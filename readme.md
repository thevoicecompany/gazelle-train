# Gazelle Trainer

This is an open source trainer for Gazelle: A Joint Speech Language Model which was made by [@hingeloss](https://github.com/stillmatic).

We walk through our own trainer and some data collation steps here.

You can find the trainer in the training folder along with a notebook walking you through the process.


- This is a preliminary release for our implementation of the training codebase for Gazelle - does not include data-preprocessing
- We are super open to collaborators and if you think you could use more code/data from us or would like to contribute to the repo - please reach out at ak@sfvoice.company
- We are training our model and it will soon be open-source!

## Data formatting

- The Data Collator takes in a HuggingFace Dataset which has to have columns "input"(text input),"audio" (list of numbers) and "text" (labels)
