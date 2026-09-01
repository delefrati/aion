<script setup lang="ts">
import { ref, nextTick } from "vue";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];  // Retrieved sources if RAG was used
}

const messages = ref<Message[]>([]);
const input = ref("");
const loading = ref(false);
const conversationId = ref<string | null>(null);
const useRag = ref(false);  // RAG toggle
const messagesEl = ref<HTMLElement | null>(null);
const inputEl = ref<HTMLTextAreaElement | null>(null);

// Advanced generation options — left blank means "use backend default".
const showAdvanced = ref(false);
const advanced = ref({
  temperature: null as number | null,
  topK: null as number | null,
  topP: null as number | null,
  repetitionPenalty: null as number | null,
  maxTokens: null as number | null,
  historyTurns: null as number | null,
  ragTopK: null as number | null,
});

function resetAdvanced() {
  advanced.value = {
    temperature: null,
    topK: null,
    topP: null,
    repetitionPenalty: null,
    maxTokens: null,
    historyTurns: null,
    ragTopK: null,
  };
}

function onKeydown(e: KeyboardEvent) {
  // Enter sends; Shift+Enter inserts a newline.
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
}

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
        use_rag: useRag.value,  // Include RAG flag
        rag_top_k: advanced.value.ragTopK,
        temperature: advanced.value.temperature,
        top_k: advanced.value.topK,
        top_p: advanced.value.topP,
        repetition_penalty: advanced.value.repetitionPenalty,
        max_tokens: advanced.value.maxTokens,
        history_turns: advanced.value.historyTurns,
      }),
    });

    messages.value.push({ role: "assistant", content: "", sources: undefined });
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
          } else if (prevEvent === "sources") {
            // Parse and store sources
            messages.value[idx].sources = data.split(";").filter(s => s.trim());
            prevEvent = null;
            continue;
          } else if (prevEvent === "token") {
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
        <div class="bubble">
          <div v-if="msg.sources" class="sources">
            <span class="sources-label">📚 Sources:</span>
            <div class="source-list">
              <span v-for="(source, j) in msg.sources" :key="j" class="source-badge">
                {{ source }}
              </span>
            </div>
          </div>
          {{ msg.content }}
        </div>
      </div>
      <div v-if="!messages.length" class="empty">
        Send a message to start a conversation.
      </div>
    </div>
    <form class="input-area" @submit.prevent="send">
      <div class="controls">
        <textarea
          ref="inputEl"
          v-model="input"
          placeholder="Type a message... (Shift+Enter for a new line)"
          :disabled="loading"
          rows="1"
          autofocus
          @keydown="onKeydown"
        />
        <button type="submit" :disabled="loading || !input.trim()">Send</button>
      </div>
      <div class="rag-toggle">
        <label>
          <input v-model="useRag" type="checkbox" />
          <span>Use RAG</span>
        </label>
        <button type="button" class="advanced-toggle" @click="showAdvanced = !showAdvanced">
          {{ showAdvanced ? "Hide" : "Show" }} advanced options
        </button>
      </div>
      <div v-if="showAdvanced" class="advanced-panel">
        <div class="advanced-grid">
          <label class="advanced-field">
            <span>Temperature</span>
            <input v-model.number="advanced.temperature" type="number" step="0.05" min="0" max="2" placeholder="default" />
          </label>
          <label class="advanced-field">
            <span>Top-K</span>
            <input v-model.number="advanced.topK" type="number" step="1" min="0" placeholder="default" />
          </label>
          <label class="advanced-field">
            <span>Top-P</span>
            <input v-model.number="advanced.topP" type="number" step="0.05" min="0" max="1" placeholder="default" />
          </label>
          <label class="advanced-field">
            <span>Repetition penalty</span>
            <input v-model.number="advanced.repetitionPenalty" type="number" step="0.05" min="1" placeholder="default" />
          </label>
          <label class="advanced-field">
            <span>Max tokens</span>
            <input v-model.number="advanced.maxTokens" type="number" step="1" min="1" placeholder="default" />
          </label>
          <label class="advanced-field">
            <span>History turns</span>
            <input v-model.number="advanced.historyTurns" type="number" step="1" min="0" placeholder="default" />
          </label>
          <label class="advanced-field">
            <span>RAG top-K</span>
            <input v-model.number="advanced.ragTopK" type="number" step="1" min="1" placeholder="default" />
          </label>
        </div>
        <button type="button" class="advanced-reset" @click="resetAdvanced">Reset to defaults</button>
      </div>
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
.sources {
  font-size: 0.85rem;
  color: #b0b0b0;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.sources-label {
  font-weight: 600;
  display: block;
  margin-bottom: 0.25rem;
}
.source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}
.source-badge {
  background: rgba(100, 200, 255, 0.2);
  border: 1px solid rgba(100, 200, 255, 0.4);
  padding: 0.2rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
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
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  border-top: 1px solid #222;
}
.controls {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
}
.rag-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 0.25rem;
}
.rag-toggle label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-size: 0.9rem;
  color: #888;
  user-select: none;
}
.rag-toggle input[type="checkbox"] {
  width: 1.2rem;
  height: 1.2rem;
  cursor: pointer;
  accent-color: #8ab4f8;
}
.advanced-toggle {
  background: none;
  border: none;
  color: #8ab4f8;
  font-size: 0.85rem;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
}
.advanced-toggle:hover {
  text-decoration: underline;
}
.advanced-panel {
  border: 1px solid #2a2a2a;
  border-radius: 0.5rem;
  padding: 0.75rem;
  background: #161616;
}
.advanced-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
}
.advanced-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.8rem;
  color: #999;
}
.advanced-field input {
  padding: 0.4rem 0.6rem;
  border-radius: 0.4rem;
  border: 1px solid #333;
  background: #1a1a1a;
  color: #e0e0e0;
  font-size: 0.9rem;
  outline: none;
}
.advanced-field input:focus {
  border-color: #8ab4f8;
}
.advanced-reset {
  margin-top: 0.75rem;
  background: none;
  border: 1px solid #333;
  border-radius: 0.4rem;
  color: #999;
  font-size: 0.8rem;
  padding: 0.3rem 0.7rem;
  cursor: pointer;
}
.advanced-reset:hover {
  border-color: #8ab4f8;
  color: #8ab4f8;
}
textarea {
  flex: 1;
  padding: 0.6rem 1rem;
  border-radius: 0.5rem;
  border: 1px solid #333;
  background: #1a1a1a;
  color: #e0e0e0;
  font-size: 1rem;
  font-family: inherit;
  outline: none;
  resize: none;
  max-height: 8rem;
  overflow-y: auto;
  line-height: 1.5;
}
textarea:focus {
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
