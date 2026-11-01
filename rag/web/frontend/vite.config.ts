import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vitest/config';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      port: 3000,
      // Proxy all /api/* requests to Flask backend (port 5000)
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:5000',
          changeOrigin: true,
        },
      },
      hmr: process.env.DISABLE_HMR !== 'true',
    },
    test: {
      // jsdom, because the tests render the real App; the suite lives outside
      // src/ so the structural gates that scan src/ (i18n keys read, api
      // members called) never count a test as a consumer.
      environment: 'jsdom',
      include: ['tests/**/*.test.{ts,tsx}'],
      setupFiles: ['tests/setup.ts'],
    },
    build: {
      // Output: rag/web/zip/dist/ — Flask serves it
      outDir: 'dist',
      emptyOutDir: true,
    },
  };

});
