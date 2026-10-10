# P6 - a skill library that improves without touching the weights

**Status: not started.** Written with Chapter 12.
Book chapter: Chapter 12 (skills and self-improvement), project 6 in the book's appendix.
Compute tier: ✅ - a small local model plus a retrieval index; no training at all.

## What you build

A loop that turns repeated failure into a reusable skill, and proves it did:

```
task family -> attempt -> failure trace -> candidate skill -> validation -> library
      ^                                                                        |
      +----------------------- next attempt retrieves it ----------------------+
```

## Why it is measured and not asserted

This project is where the book's "a number needs a label" rule earns its keep. Two
measurements have to exist before any claim of self-improvement is meaningful:

* **retrieval**: does the right skill get found? Report it the way the chapter does -
  recall at 1, against a plain keyword baseline and against an embedding baseline. Without
  a baseline, "our router works" is an opinion.
* **outcome**: does the retrieved skill change what happens on the task? Report success
  rate on the *same* task families, with and without the library populated.

Both numbers are `measured`. The baseline is `measured` too. Anything read off a paper is
`reported` and stays labelled that way.

## The task set is generated here, not downloaded

`datasets/manifest.yaml` lists it under `derived`: 10 recurring task families with 20
instances each. The recurrence is the point - a skill library is only interesting when the
same shape of problem comes back, and no public dataset is built to make that happen on
purpose.

## The four questions this project answers

1. When does a skill belong in the library rather than in the prompt?
2. What does a validation step have to be, so that a bad skill cannot get in?
3. What happens to retrieval when there are a hundred skills rather than five?
4. What is the *evidence* that the library helped - and what would have looked like
   evidence while being nothing of the kind?

## What is not here yet

- The generated task families and their instance generator.
- The skill library format, its validation step, and the retrieval baselines.
