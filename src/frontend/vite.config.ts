import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3900,
    proxy: {
      "/api": {
        target: "http://localhost:8900",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
