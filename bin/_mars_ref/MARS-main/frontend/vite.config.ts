import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3003,
    host: '0.0.0.0',
    proxy: {
      '/matagent': {
        target: 'http://159.75.91.126:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/matagent/, '/matagent')
      }
    }
  },
})
