import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,           // listen on all interfaces (important for tunneling)
    strictPort: true,     // don’t auto-switch ports
    allowedHosts: [
      '.trycloudflare.com'   // allow any *.trycloudflare.com host
    ],
    hmr: {
      clientPort: 443     // needed for HMR when using HTTPS tunnel
    }
  }
})
