<script setup lang="ts">
import { ref } from "vue";
import Chat from "./components/Chat.vue";
import DocumentManager from "./components/DocumentManager.vue";

const view = ref<"chat" | "documents">("chat");
</script>

<template>
  <div class="app">
    <header>
      <h1>AION</h1>
      <nav class="nav-tabs">
        <button
          :class="{ active: view === 'chat' }"
          @click="view = 'chat'"
          class="nav-tab"
        >
          💬 Chat
        </button>
        <button
          :class="{ active: view === 'documents' }"
          @click="view = 'documents'"
          class="nav-tab"
        >
          📄 Documents
        </button>
      </nav>
    </header>
    <main>
      <Chat v-show="view === 'chat'" />
      <DocumentManager v-show="view === 'documents'" />
    </main>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: #0a0a0a;
  color: #e0e0e0;
}
.app {
  max-width: 800px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
}
header {
  padding: 1rem;
  border-bottom: 1px solid #222;
}
header h1 {
  font-size: 1.2rem;
  font-weight: 600;
  color: #8ab4f8;
  margin-bottom: 1rem;
}
.nav-tabs {
  display: flex;
  gap: 0.5rem;
}
.nav-tab {
  padding: 0.5rem 1rem;
  background: transparent;
  border: 1px solid #333;
  border-bottom: none;
  color: #999;
  cursor: pointer;
  border-radius: 0.5rem 0.5rem 0 0;
  transition: all 0.2s;
  font-size: 0.95rem;
}
.nav-tab:hover {
  color: #b0b0b0;
  border-color: #444;
}
.nav-tab.active {
  background: #1a1a1a;
  color: #e0e0e0;
  border-color: #0066cc;
}
main {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
</style>
