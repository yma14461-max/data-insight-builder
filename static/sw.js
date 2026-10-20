// AI Data Insight Builder Service Worker
const CACHE_NAME = 'data-insight-v1';
const STATIC_ASSETS = [
  '/',
  '/static/style.css',
  '/static/app.js',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // API 호출이나 POST 요청은 캐시하지 않고 항상 네트워크로 통과
  if (event.request.method !== 'GET' || url.pathname.startsWith('/inspect') || url.pathname.startsWith('/suggest') || url.pathname.startsWith('/run') || url.pathname.startsWith('/explain') || url.pathname.startsWith('/ask') || url.pathname.startsWith('/query') || url.pathname.startsWith('/auth')) {
    return;
  }

  event.respondWith(
    fetch(event.request).catch(() => {
      return caches.match(event.request);
    })
  );
});
