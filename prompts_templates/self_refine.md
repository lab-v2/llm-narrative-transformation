# Self-Refine baseline prompts

Three stateless calls per story: (1) zero-shot transform → (2) critique → (3) refine.
Exactly one refinement round. All three use the system prompt `You are a helpful assistant.`
The refine output is wrapped in `<rewritten_story>` tags (parsed out downstream).

---

**Call 1 — Zero-shot transform**  (input: original story)

```
Make the following story more {target}:

{story}
```

---

**Call 2 — Critique**  (input: original story + the Call-1 output)

```
Here is an original story and a version rewritten to be more {target}.

Original story:
{original}

Rewritten (more {target}) version:
{transformed}

Critique the rewritten version: which parts still read as {source}, and what specifically should change to make it more {target}? Do not rewrite the story — only give the critique.
```

---

**Call 3 — Refine**  (input: the Call-1 output + the Call-2 critique; the original is intentionally omitted)

```
Below is a story that was rewritten to be more {target}, along with a critique. Rewrite it again to be more {target}, addressing the critique.

Story:
{transformed}

Critique:
{critique}

Output ONLY the rewritten story wrapped exactly in these tags, with no preamble and no explanation of your changes:
<rewritten_story>
...complete rewritten story...
</rewritten_story>
```
