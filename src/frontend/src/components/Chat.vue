<script setup lang="ts">
import { ref, nextTick } from "vue";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const messages = ref<Message[]>([]);
const input = ref("");
const loading = ref(false);
const conversationId = ref<string | null>(null);
const messagesEl = ref<HTMLElement | null>(null);
const inputEl = ref<HTMLInputElement | null>(null);

async function send() {
  const text = input.value.trim();
  if (!text || loading.value) return;

  messages.value.push({ role: "user", content: text });
  input.value = "";
  loading.value = true;

  await nextTick();
  messagesEl.value?.scrollTo(0, messagesEl.value.scrollHeight);

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        conversation_id: conversationId.value,
      }),
    });

    messages.value.push({ role: "assistant", content: "" });
    const idx = messages.value.length - 1;

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let prevEvent: string | null = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (let line of lines) {
        line = line.replace(/\r$/, "");
        if (line.startsWith("event: done")) {
          // next data line has the conversation id — handled below
        } else if (line.startsWith("data: ")) {
          const data = line.slice(6);
          // if previous line was "event: done", this is the conversation id
          if (prevEvent === "done") {
            conversationId.value = data;
          } else {
            messages.value[idx].content += data;
          }
          prevEvent = null;
          continue;
        }
        if (line.startsWith("event: ")) {
          prevEvent = line.slice(7);
        }
      }

      await nextTick();
      messagesEl.value?.scrollTo(0, messagesEl.value.scrollHeight);
    }
  } catch (e) {
    messages.value.push({
      role: "assistant",
      content: "Connection error. Is the backend running?",
    });
  } finally {
    loading.value = false;
    await nextTick();
    inputEl.value?.focus();
  }
}
</script>

<template>
  <div class="chat">
    <div class="messages" ref="messagesEl">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        :class="['message', msg.role]"
      >
        <div class="bubble">{{ msg.content }}</div>
      </div>
      <div v-if="!messages.length" class="empty">
        Send a message to start a conversation.
      </div>
    </div>
    <form class="input-area" @submit.prevent="send">
      <input
        ref="inputEl"
        v-model="input"
        placeholder="Type a message..."
        :disabled="loading"
        autofocus
      />
      <button type="submit" :disabled="loading || !input.trim()">Send</button>
    </form>
  </div>
</template>

<style scoped>
.chat {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.message {
  display: flex;
}
.message.user {
  justify-content: flex-end;
}
.bubble {
  max-width: 70%;
  padding: 0.6rem 1rem;
  border-radius: 1rem;
  line-height: 1.5;
  white-space: pre-wrap;
}
.user .bubble {
  background: #1a3a5c;
  border-bottom-right-radius: 0.25rem;
}
.assistant .bubble {
  background: #1e1e1e;
  border-bottom-left-radius: 0.25rem;
}
.empty {
  color: #666;
  text-align: center;
  margin-top: 40%;
}
.input-area {
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  border-top: 1px solid #222;
}
input {
  flex: 1;
  padding: 0.6rem 1rem;
  border-radius: 0.5rem;
  border: 1px solid #333;
  background: #1a1a1a;
  color: #e0e0e0;
  font-size: 1rem;
  outline: none;
}
input:focus {
  border-color: #8ab4f8;
}
button {
  padding: 0.6rem 1.2rem;
  border-radius: 0.5rem;
  border: none;
  background: #8ab4f8;
  color: #0a0a0a;
  font-weight: 600;
  cursor: pointer;
}
button:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>
