# Few-Shot baseline prompt

2 fixed exemplars per direction (expert-chosen, held out from the test set). The exemplar block
is identical for every test story; the final line is byte-for-byte the zero-shot instruction.

- **Forward (C→I) exemplars:** *Stone Soup*, *The Hunter and the Doves*
- **Inverse (I→C) exemplars:** *The Hare and the Tortoise*, *The Little Red Hen*

**System prompt**

```
You are a helpful assistant.
```

**User prompt** (`{source_cap}` / `{target_cap}` are the capitalized framing words; `{ex1_*}`,
`{ex2_*}` are the exemplar originals/transformed versions)

```
Here are examples of rewriting a story to be more {target}.

Example 1:
{source_cap} story:
{ex1_original}

{target_cap} version:
{ex1_transformed}

Example 2:
{source_cap} story:
{ex2_original}

{target_cap} version:
{ex2_transformed}

Make the following story more {target}:

{story}
```
