# Exhibitions

## Directions on Formatting

In this directory, each directory should contain a single scientific or
engineered model/project using ngc-learn. The content and organization within
an exhibit's directory is dependent on the project/model/agent the exhibit is 
about and generally will contain application/task-specific structuring code, 
e.g., agent classes with specified functions pertinent to the problem/task 
examined in a paper's experiment. 
<!--Datasets should not be stored in this repo but instead referenced
to their source locations.-->

The only file we specifically require in an exhibition's folder is a single
markdown file -- `README.md` -- containing information about the specific model,
agent, or system publicly offered. Specifically, this markdown file should
start with:
```markdown
# <NAME_OF_EXHIBITION_MODEL>

<b>Version</b>: ngclearn==X.Y.Z, ngcsimlib==A.B.C

...rest of text...
```

It is recommended to include useful instructions
for how to run the code to reproduce particular experiments or simulation studies that
the model/agent/system is meant to be used within. Documenting constrained
functionality of the model/agent/system is further encouraged.
