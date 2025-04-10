import os
import json
import logging
import time
import random
import requests
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from requests.exceptions import SSLError, HTTPError
from datetime import datetime
import aiohttp
import asyncio

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class SERPResult:
    """Structure pour un résultat SERP"""
    def __init__(self, url: str, rank: int, meta_title: str, meta_description: str, domain: str):
        self.url = url
        self.rank = rank
        self.meta_title = meta_title
        self.meta_description = meta_description
        self.domain = domain
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire"""
        return {
            'url': self.url,
            'rank': self.rank,
            'meta_title': self.meta_title,
            'meta_description': self.meta_description,
            'domain': self.domain
        }

class URLContent:
    """Structure pour le contenu d'une URL"""
    def __init__(self, url: str, rank: int):
        self.url = url
        self.rank = rank
        self.h1 = ""
        self.introduction = ""
        self.structure_headings = []
        self.full_content = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire"""
        return {
            'url': self.url,
            'rank': self.rank,
            'h1': self.h1,
            'introduction': self.introduction,
            'structure_hn': self.structure_headings,
            'full_content': self.full_content
        }

class SERPScraper:
    """
    Scraper pour extraire les données SERP et le contenu des URLs concurrentes.
    Utilise Bright Data pour accéder aux résultats Google.
    """
    
    def __init__(self, content_engine=None, data_storage=None):
        """
        Initialise le scraper.
        
        Args:
            content_engine: Moteur de contenu (optionnel)
            data_storage: Gestionnaire de stockage des données (optionnel)
        """
        self.content_engine = content_engine
        self.data_storage = data_storage
        
        # Initialisation du cache de résultats SERP
        self.cache_dir = os.path.join("cache", "serp")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Durée de validité du cache en jours
        self.cache_validity_days = int(os.getenv('SERP_CACHE_VALIDITY_DAYS', '30'))
        
        # Chargement des variables d'environnement pour Bright Data
        self.bright_config = {
            'username': os.getenv('BRIGHTDATA_USERNAME', ''),
            'password': os.getenv('BRIGHTDATA_PASSWORD', ''),
            'host': os.getenv('BRIGHTDATA_HOST', ''),
            'port': os.getenv('BRIGHTDATA_PORT', ''),
            'country': os.getenv('BRIGHTDATA_COUNTRY', 'fr'),
            'render_js': os.getenv('BRIGHTDATA_RENDER_JS', 'false').lower() == 'true'
        }
        
        # Vérification des variables d'environnement requises
        required_vars = ['username', 'password', 'host', 'port']
        missing_vars = [var for var in required_vars if not self.bright_config[var]]
        if missing_vars:
            logger.warning(f"Configuration Bright Data incomplète: {', '.join(missing_vars)} manquant(s)")
        
        # Paramètres de scraping
        self.settings = {
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'timeout': int(os.getenv('TIMEOUT', '30')),
            'delay_min': float(os.getenv('DELAY_MIN', '2')),
            'delay_max': float(os.getenv('DELAY_MAX', '5'))
        }
        
        # Configuration des sessions
        self.ua = UserAgent()
        self.session = self._create_bright_data_session()
        
        # Statistiques de scraping
        self.stats = {
            'total_keywords': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_urls': 0,
            'url_successful': 0,
            'url_failed': 0,
            'errors': []
        }
    
    def _create_bright_data_session(self) -> requests.Session:
        """
        Crée une session HTTP configurée avec le proxy Bright Data.
        
        Returns:
            Session HTTP configurée
        """
        # Vérification des informations de configuration
        if not all([self.bright_config['username'], self.bright_config['password'], 
                  self.bright_config['host'], self.bright_config['port']]):
            logger.warning("Configuration Bright Data incomplète, le scraping SERP pourrait ne pas fonctionner")
        
        # Construction de l'URL du proxy
        proxy_url = (
            f"http://{self.bright_config['username']}:"
            f"{self.bright_config['password']}@"
            f"{self.bright_config['host']}:"
            f"{self.bright_config['port']}"
        )
        
        # Création et configuration de la session
        session = requests.Session()
        session.verify = os.getenv('VERIFY_SSL', 'True').lower() != 'false'
        session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        
        # Désactivation des avertissements SSL si demandé
        if os.getenv('SUPPRESS_SSL_WARNINGS', 'False').lower() == 'true':
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Génère des en-têtes HTTP aléatoires pour éviter la détection.
        
        Returns:
            Dictionnaire d'en-têtes HTTP
        """
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': f"{self.bright_config['country']}-{self.bright_config['country'].upper()},{self.bright_config['country']};q=0.9,en-US;q=0.8,en;q=0.7",
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
    
    def _get_search_url(self, keyword: str) -> str:
        """
        Construit l'URL de recherche Google.
        
        Args:
            keyword: Mot-clé à rechercher
            
        Returns:
            URL de recherche Google
        """
        params = {
            'q': keyword,
            'num': 10,  # Nombre de résultats par page
            'hl': self.bright_config['country'],
            'gl': self.bright_config['country'].upper(),
            'pws': '0'  # Désactive la personnalisation des résultats
        }
        return f"https://www.google.com/search?{urlencode(params)}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _scrape_serp(self, keyword: str) -> List[SERPResult]:
        """
        Scrape les résultats SERP pour un mot-clé.
        
        Args:
            keyword: Mot-clé à rechercher
            
        Returns:
            Liste des résultats SERP
            
        Raises:
            Exception: En cas d'erreur de scraping ou de détection anti-bot
        """
        try:
            url = self._get_search_url(keyword)
            
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.settings['timeout']
            )
            response.raise_for_status()
            
            # Vérification anti-bot
            if any(marker in response.text.lower() for marker in ['captcha', 'unusual traffic']):
                logger.error(f"Anti-bot detection triggered for keyword: {keyword}")
                raise Exception("Détection anti-bot par Google")
            
            # Extraction des résultats
            return self._extract_serp_results(response.text)
        except Exception as e:
            logger.error(f"Erreur lors du scraping SERP pour '{keyword}': {str(e)}")
            raise
    
    def _extract_serp_results(self, html: str) -> List[SERPResult]:
        """
        Extrait les résultats SERP du HTML.
        
        Args:
            html: Contenu HTML de la page de résultats
            
        Returns:
            Liste des résultats SERP
        """
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        rank = 1
        
        # Sélecteur pour les résultats organiques
        for result_div in soup.select('div.g'):
            try:
                # Extraction des éléments
                title_element = result_div.select_one('h3')
                url_element = result_div.select_one('a')
                snippet_element = result_div.select_one('div.VwiC3b')
                
                if not (title_element and url_element):
                    continue
                
                url = url_element.get('href', '').strip()
                
                # Ignorer les URLs non HTTP
                if not url.startswith('http'):
                    continue
                
                # Extraire le domaine
                domain = url.split('/')[2] if len(url.split('/')) > 2 else url
                
                # Créer le résultat
                serp_result = SERPResult(
                    url=url,
                    rank=rank,
                    meta_title=title_element.get_text(strip=True),
                    meta_description=snippet_element.get_text(strip=True) if snippet_element else '',
                    domain=domain
                )
                
                results.append(serp_result)
                rank += 1
            except Exception as e:
                logger.debug(f"Erreur d'extraction d'un résultat SERP: {str(e)}")
                continue
        
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _scrape_url_content(self, url: str, rank: int) -> URLContent:
        """
        Scrape le contenu d'une URL concurrente.
        
        Args:
            url: URL à scraper
            rank: Rang de l'URL dans les résultats SERP
            
        Returns:
            Contenu de l'URL
            
        Raises:
            Exception: En cas d'erreur de scraping
        """
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.settings['timeout'],
                verify=os.getenv('VERIFY_SSL', 'True').lower() != 'false'
            )
            response.raise_for_status()
            
            return self._extract_url_content(response.text, url, rank)
        except Exception as e:
            logger.error(f"Erreur lors du scraping de l'URL {url}: {str(e)}")
            
            # Essai sans vérification SSL en cas d'erreur SSL
            if isinstance(e, SSLError):
                try:
                    logger.info(f"Nouvel essai sans vérification SSL pour {url}")
                    response = requests.get(
                        url,
                        headers=self._get_headers(),
                        timeout=self.settings['timeout'],
                        verify=False
                    )
                    response.raise_for_status()
                    
                    return self._extract_url_content(response.text, url, rank)
                except Exception as inner_e:
                    logger.error(f"Échec persistant pour {url}: {str(inner_e)}")
            
            raise
    
    def _extract_url_content(self, html: str, url: str, rank: int) -> URLContent:
        """
        Extrait le contenu structuré d'une page HTML.
        
        Args:
            html: Contenu HTML de la page
            url: URL de la page
            rank: Rang de l'URL dans les résultats SERP
            
        Returns:
            Contenu structuré de l'URL
        """
        soup = BeautifulSoup(html, 'html.parser')
        url_content = URLContent(url, rank)
        
        # Extraction du H1
        h1_element = soup.find('h1')
        if h1_element:
            url_content.h1 = h1_element.get_text(strip=True)
        
        # Extraction de l'introduction (premiers paragraphes)
        introduction = []
        main_content = self._find_main_content(soup)
        
        if main_content:
            paragraphs = main_content.find_all('p', limit=3)
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Ignorer les paragraphes trop courts
                    introduction.append(text)
        
        url_content.introduction = ' '.join(introduction)
        
        # Extraction de la structure des titres
        url_content.structure_headings = self._extract_heading_structure(soup)
        
        # Extraction du contenu complet
        content_elements = []
        if main_content:
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol']):
                text = element.get_text(strip=True)
                if text:
                    content_elements.append(text)
        
        url_content.full_content = ' '.join(content_elements)
        
        return url_content
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """
        Identifie le contenu principal de la page.
        
        Args:
            soup: Objet BeautifulSoup de la page
            
        Returns:
            Élément BeautifulSoup contenant le contenu principal ou None
        """
        # Liste des sélecteurs potentiels pour le contenu principal
        selectors = [
            'article', 'main', '#content', '.content', '#main-content',
            '.main-content', '.post-content', '.entry-content',
            '[role="main"]', '.article-content', '.entry', '.post'
        ]
        
        # Essai des sélecteurs
        for selector in selectors:
            try:
                content = soup.select_one(selector)
                if content and len(content.get_text(strip=True)) > 200:
                    return content
            except Exception:
                continue
        
        # Tentative de trouver l'élément avec le plus de texte
        candidates = soup.find_all(['article', 'main', 'div', 'section'])
        if candidates:
            try:
                return max(
                    [c for c in candidates if len(c.get_text(strip=True)) > 200],
                    key=lambda x: len(x.get_text(strip=True)),
                    default=None
                )
            except Exception:
                pass
        
        return None
    
    def _extract_heading_structure(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extrait la structure hiérarchique des titres.
        
        Args:
            soup: Objet BeautifulSoup de la page
            
        Returns:
            Structure hiérarchique des titres
        """
        headings = []
        current_h2 = None
        current_h3 = None
        
        # Parcours des éléments de titre
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            heading_text = heading.get_text(strip=True)
            
            if not heading_text:
                continue
            
            if heading.name == 'h2':
                current_h2 = {'h2': heading_text, 'h3_subtitles': []}
                headings.append(current_h2)
                current_h3 = None
            
            elif heading.name == 'h3' and current_h2:
                current_h3 = {'h3': heading_text, 'h4_subtitles': []}
                current_h2['h3_subtitles'].append(current_h3)
            
            elif heading.name == 'h4' and current_h3:
                current_h3['h4_subtitles'].append({'h4': heading_text})
        
        return headings
    
    async def _scrape_url_content_async(self, url: str, rank: int, session: aiohttp.ClientSession) -> URLContent:
        """
        Version asynchrone du scraping de contenu d'URL.
        
        Args:
            url: URL à scraper
            rank: Rang de l'URL dans les résultats SERP
            session: Session aiohttp
            
        Returns:
            Contenu de l'URL
            
        Raises:
            Exception: En cas d'erreur de scraping
        """
        try:
            headers = self._get_headers()
            async with session.get(url, headers=headers, timeout=self.settings['timeout'], 
                               ssl=None if os.getenv('VERIFY_SSL', 'True').lower() == 'false' else True) as response:
                response.raise_for_status()
                html = await response.text()
                
                return self._extract_url_content(html, url, rank)
        except Exception as e:
            logger.error(f"Erreur lors du scraping async de l'URL {url}: {str(e)}")
            
            # Essai sans vérification SSL en cas d'erreur SSL
            if "SSL" in str(e):
                try:
                    logger.info(f"Nouvel essai sans vérification SSL pour {url}")
                    async with session.get(url, headers=headers, timeout=self.settings['timeout'], ssl=False) as response:
                        response.raise_for_status()
                        html = await response.text()
                        
                        return self._extract_url_content(html, url, rank)
                except Exception as inner_e:
                    logger.error(f"Échec persistant pour {url}: {str(inner_e)}")
            
            raise
    
    async def scrape_keyword_with_content(self, keyword: str, state_manager=None, force_refresh=False) -> Dict[str, Any]:
        """
        Scrape un mot-clé et le contenu des URLs concurrentes en parallèle.
        
        Args:
            keyword: Mot-clé à scraper
            state_manager: Gestionnaire d'état (optionnel)
            force_refresh: Force le rafraîchissement du cache même s'il est valide
            
        Returns:
            Résultats du scraping
        """
        try:
            # Vérifier le cache d'abord si on ne force pas le rafraîchissement
            if not force_refresh:
                cached_results = self._load_from_cache(keyword)
                if cached_results:
                    logger.info(f"Résultats chargés depuis le cache pour '{keyword}'")
                    
                    if state_manager:
                        state_manager.set("scraping_status", {
                            "step": "completed",
                            "keyword": keyword,
                            "status": "completed",
                            "from_cache": True
                        })
                        state_manager.set("scraping_results", cached_results)
                    
                    self.stats['total_keywords'] += 1
                    self.stats['processed'] += 1
                    self.stats['successful'] += 1
                    
                    return cached_results
            
            # Si pas de cache valide ou rafraîchissement forcé, procéder au scraping
            logger.info(f"Début du scraping pour '{keyword}'")
            self.stats['total_keywords'] += 1
            self.stats['processed'] += 1
            
            # Étape 1: Scraping SERP
            if state_manager:
                state_manager.set("scraping_status", {
                    "step": "serp",
                    "keyword": keyword,
                    "status": "in_progress"
                })
            
            serp_results = self._scrape_serp(keyword)
            
            if not serp_results:
                raise ValueError(f"Aucun résultat SERP trouvé pour '{keyword}'")
            
            # Conversion des résultats SERP en dictionnaires
            serp_data = [result.to_dict() for result in serp_results]
            
            # Mise à jour des données si demandé
            if self.data_storage:
                self.data_storage.save_serp_results(keyword, serp_data)
            
            # Étape 2: Scraping du contenu des URLs en parallèle
            url_count = len(serp_results)
            self.stats['total_urls'] += url_count
            
            # Créer une session aiohttp réutilisable pour toutes les requêtes
            async with aiohttp.ClientSession() as session:
                # Limiter la concurrence pour éviter de surcharger les serveurs
                semaphore = asyncio.Semaphore(5)  # Maximum 5 requêtes simultanées
                
                async def fetch_with_semaphore(result, index):
                    """Fonction wrapper pour gérer le semaphore et le statut"""
                    async with semaphore:
                        if state_manager:
                            state_manager.set("scraping_status", {
                                "step": "url_content",
                                "keyword": keyword,
                                "url": result.url,
                                "progress": f"{index+1}/{url_count}",
                                "status": "in_progress"
                            })
                        
                        try:
                            logger.info(f"Scraping URL ({index+1}/{url_count}): {result.url}")
                            url_content = await self._scrape_url_content_async(result.url, result.rank, session)
                            self.stats['url_successful'] += 1
                            
                            # Délai court entre les requêtes pour éviter la détection
                            await asyncio.sleep(random.uniform(0.5, 1.0))
                            
                            return url_content.to_dict()
                        except Exception as e:
                            logger.error(f"Erreur lors du scraping de {result.url}: {str(e)}")
                            self.stats['url_failed'] += 1
                            self.stats['errors'].append({
                                'keyword': keyword,
                                'url': result.url,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            })
                            return None
                
                # Créer les tâches pour chaque URL à scraper
                tasks = [
                    fetch_with_semaphore(result, i) 
                    for i, result in enumerate(serp_results)
                ]
                
                # Exécuter toutes les tâches en parallèle et récupérer les résultats
                results = await asyncio.gather(*tasks)
                
                # Filtrer les résultats None (erreurs)
                urls_content = [r for r in results if r is not None]
            
            # Mise à jour des données si demandé
            if self.data_storage:
                self.data_storage.save_urls_content(keyword, urls_content)
            
            # Préparation des résultats
            results = {
                'keyword': keyword,
                'serp_results': serp_data,
                'urls_content': urls_content,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Sauvegarder dans le cache
            self._save_to_cache(keyword, results)
            
            logger.info(f"Scraping terminé pour '{keyword}': {len(serp_data)} résultats, {len(urls_content)} contenus URL")
            self.stats['successful'] += 1
            
            if state_manager:
                state_manager.set("scraping_status", {
                    "step": "completed",
                    "keyword": keyword,
                    "status": "completed",
                    "cached": True
                })
                state_manager.set("scraping_results", results)
            
            return results
        except Exception as e:
            logger.error(f"Erreur lors du scraping de '{keyword}': {str(e)}")
            self.stats['failed'] += 1
            self.stats['errors'].append({
                'keyword': keyword,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            if state_manager:
                state_manager.set("scraping_status", {
                    "step": "error",
                    "keyword": keyword,
                    "error": str(e),
                    "status": "error"
                })
            
            return {
                'keyword': keyword,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_cache_path(self, keyword: str) -> str:
        """
        Obtient le chemin du fichier de cache pour un mot-clé donné.
        
        Args:
            keyword: Mot-clé à rechercher
            
        Returns:
            Chemin du fichier de cache
        """
        # Normaliser le mot-clé pour utilisation comme nom de fichier
        keyword_normalized = keyword.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{keyword_normalized}.json")
    
    def _is_cache_valid(self, keyword: str) -> bool:
        """
        Vérifie si un résultat en cache est valide (existe et n'est pas expiré).
        
        Args:
            keyword: Mot-clé à vérifier
            
        Returns:
            True si le cache est valide, False sinon
        """
        cache_path = self._get_cache_path(keyword)
        
        # Vérifier si le fichier existe
        if not os.path.exists(cache_path):
            return False
        
        # Vérifier la date de dernière modification
        from datetime import datetime, timedelta
        
        file_timestamp = os.path.getmtime(cache_path)
        file_date = datetime.fromtimestamp(file_timestamp)
        
        # Vérifier si le cache a expiré
        expiration_date = datetime.now() - timedelta(days=self.cache_validity_days)
        return file_date > expiration_date
    
    def _load_from_cache(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        Charge les résultats depuis le cache pour un mot-clé donné.
        
        Args:
            keyword: Mot-clé à rechercher
            
        Returns:
            Résultats en cache ou None si le cache n'est pas valide
        """
        if not self._is_cache_valid(keyword):
            return None
        
        cache_path = self._get_cache_path(keyword)
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Erreur lors du chargement depuis le cache pour '{keyword}': {str(e)}")
            return None
    
    def _save_to_cache(self, keyword: str, data: Dict[str, Any]) -> bool:
        """
        Sauvegarde les résultats dans le cache pour un mot-clé donné.
        
        Args:
            keyword: Mot-clé associé aux résultats
            data: Résultats à sauvegarder
            
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        cache_path = self._get_cache_path(keyword)
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde dans le cache pour '{keyword}': {str(e)}")
            return False
    
    def extract_competitor_data(self, serp_results: List[Dict[str, Any]], urls_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extrait et organise les données concurrentielles complètes pour le prompt.
        
        Args:
            serp_results: Liste des résultats SERP
            urls_content: Liste des contenus d'URL
            
        Returns:
            Données concurrentielles organisées pour une analyse approfondie
        """
        data = {
            'metatitles': [],
            'metadescriptions': [],
            'h1s': [],
            'h2s': [],
            'content_structures': [],
            'full_content_analysis': [],
            'intent_keywords': set(),
            'search_data': {
                'top_domains': [],
                'content_length_avg': 0,
                'most_common_sections': []
            }
        }
        
        # Dictionnaires pour l'analyse des données
        domain_counts = {}
        all_h2s = []
        all_keywords = []
        total_content_length = 0
        
        # Extraire les méta-informations des résultats SERP
        for result in serp_results:
            # Ajouter les méta-informations
            data['metatitles'].append({
                'title': result.get('meta_title', ''),
                'rank': result.get('rank', 0),
                'domain': result.get('domain', '')
            })
            
            data['metadescriptions'].append({
                'description': result.get('meta_description', ''),
                'rank': result.get('rank', 0),
                'domain': result.get('domain', '')
            })
            
            # Compter les domaines pour l'analyse de concurrence
            domain = result.get('domain', '')
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
            # Extraire des mots-clés potentiels d'intention de recherche
            title = result.get('meta_title', '').lower()
            description = result.get('meta_description', '').lower()
            all_text = f"{title} {description}"
            
            # Mots-clés à exclure car trop génériques
            stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'pour', 'avec', 'dans', 'sur', 'en', 'et', 'ou', 'qui', 'que', 'quoi', 'comment', 'par'}
            words = [w for w in all_text.split() if len(w) > 3 and w not in stopwords]
            all_keywords.extend(words)
        
        # Analyse des contenus d'URL
        h2_counter = {}
        
        for content in urls_content:
            # Ajouter les H1
            data['h1s'].append({
                'h1': content.get('h1', ''),
                'rank': content.get('rank', 0),
                'url': content.get('url', '')
            })
            
            # Analyser les titres et la structure
            structure = content.get('structure_hn', [])
            
            # Extraire et compter les H2 pour analyse de fréquence
            h2s = [h2_item.get('h2', '') for h2_item in structure if 'h2' in h2_item]
            all_h2s.extend(h2s)
            
            for h2 in h2s:
                data['h2s'].append({
                    'h2': h2,
                    'rank': content.get('rank', 0),
                    'url': content.get('url', '')
                })
                h2_counter[h2] = h2_counter.get(h2, 0) + 1
            
            # Ajouter la structure complète avec H2 et H3 imbriqués
            data['content_structures'].append({
                'rank': content.get('rank', 0),
                'url': content.get('url', ''),
                'structure': structure
            })
            
            # Analyser le contenu complet
            full_content = content.get('full_content', '')
            
            if full_content:
                content_length = len(full_content.split())
                total_content_length += content_length
                
                # Ajouter une analyse du contenu complet
                intro = content.get('introduction', '')
                
                data['full_content_analysis'].append({
                    'rank': content.get('rank', 0),
                    'url': content.get('url', ''),
                    'h1': content.get('h1', ''),
                    'content_length': content_length,
                    'intro_snippet': intro[:200] + '...' if len(intro) > 200 else intro,
                    'content_snippet': full_content[:500] + '...' if len(full_content) > 500 else full_content
                })
        
        # Déterminer les mots-clés d'intention de recherche les plus fréquents
        from collections import Counter
        keyword_counter = Counter(all_keywords)
        
        # Trouver les 15 mots-clés les plus fréquents comme indicateurs d'intention
        for word, count in keyword_counter.most_common(15):
            if count > 1 and len(word) > 3:  # Éviter les mots peu fréquents ou trop courts
                data['intent_keywords'].add(word)
        
        # Calculer la longueur moyenne du contenu
        if urls_content:
            data['search_data']['content_length_avg'] = total_content_length // len(urls_content)
        
        # Trouver les domaines les plus fréquents dans les résultats
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        data['search_data']['top_domains'] = [{'domain': domain, 'count': count} for domain, count in top_domains]
        
        # Trouver les sections (H2) les plus communes à travers les concurrents
        top_sections = sorted(h2_counter.items(), key=lambda x: x[1], reverse=True)
        data['search_data']['most_common_sections'] = [{'title': title, 'frequency': count} for title, count in top_sections if count > 1]
        
        # Convertir les ensembles en listes pour la sérialisation JSON
        data['intent_keywords'] = list(data['intent_keywords'])
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de scraping.
        
        Returns:
            Statistiques de scraping
        """
        return {
            'total_keywords': self.stats['total_keywords'],
            'processed': self.stats['processed'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'success_rate': f"{(self.stats['successful'] / self.stats['total_keywords']) * 100:.1f}%" if self.stats['total_keywords'] > 0 else "0%",
            'total_urls': self.stats['total_urls'],
            'url_successful': self.stats['url_successful'],
            'url_failed': self.stats['url_failed'],
            'url_success_rate': f"{(self.stats['url_successful'] / self.stats['total_urls']) * 100:.1f}%" if self.stats['total_urls'] > 0 else "0%",
            'errors_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else []
        }
    
    def reset_stats(self) -> None:
        """Réinitialise les statistiques de scraping."""
        self.stats = {
            'total_keywords': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_urls': 0,
            'url_successful': 0,
            'url_failed': 0,
            'errors': []
        }
