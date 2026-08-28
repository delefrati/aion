<script setup lang="ts">
import Chat from "./components/Chat.vue";
</script>

<template>
  <div class="app">
    <header>
      <h1>AION</h1>
    </header>
    <main>
      <Chat />
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
}
main {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
</style>
