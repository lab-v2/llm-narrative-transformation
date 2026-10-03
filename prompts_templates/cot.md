# Chain-of-Thought (CoT) baseline prompt

The model reasons step-by-step, then emits the final story inside `<rewritten_story>` tags
(parsed out downstream so only the story is scored).

**System prompt**

```
You are an expert creative writer and literary analyst.
```

**User prompt**

```
Your task is to rewrite the story below so that it reads as more {target} (currently it reads as more {source}).

Think step by step before writing the new version:

Step 1 — Analyze: Identify the specific elements that make the story {source} — e.g. the characters' goals and motivations, the source of conflict, how key decisions are made, how relationships and values are portrayed, and how it resolves.

Step 2 — Plan: For each element you identified, state concretely how you will reframe it to be more {target}.

Step 3 — Rewrite: Apply your plan and produce the full rewritten story.

Show your reasoning for Steps 1 and 2, then output ONLY the final rewritten story wrapped exactly in these tags:
<rewritten_story>
...complete rewritten story...
</rewritten_story>

Story:
{story}
```
