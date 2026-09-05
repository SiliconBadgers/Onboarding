# Stage 1 — PyTorch model

**Goal:** know which operations the chip must perform, and why every number ends up an
integer.

## A 3x3 example, worked all the way through

Say this is the image we are given:

```
[ 0 1 0 ]
[ 0 1 0 ]
[ 0 1 0 ]
```

Looking at it, it seems to be a 1.

Now imagine our PyTorch model has seen 10,000 3x3 images that resemble the number one.
The 1s would generally sit in the same-ish area, right? That is what a **weight** is: a
number saying how much a given pixel matters.

### Layer 1, filter one: the vertical line

Say this is our "vertical" weight matrix:

```
[ -0.5   2.0  -0.5 ]
[ -0.5   2.0  -0.5 ]
[ -0.5   2.0  -0.5 ]
```

The first job is to multiply each row against the matching row of the image and add up
the products — a dot product.

```
Row 1: (0 * -0.5) + (1 * 2.0) + (0 * -0.5) = 2.0
Row 2: (0 * -0.5) + (1 * 2.0) + (0 * -0.5) = 2.0
Row 3: (0 * -0.5) + (1 * 2.0) + (0 * -0.5) = 2.0
Final score: 2.0 + 2.0 + 2.0 = 6.0
```

That is a very high score. If the weight matrix looked any different — a horizontal
line, say — the final score would be much lower.

### Layer 1, filter two: the horizontal line

Say this is our "horizontal" weight matrix:

```
[ -0.5  -0.5  -0.5 ]
[  2.0   2.0   2.0 ]
[ -0.5  -0.5  -0.5 ]
```

Same job, row by row against the same image:

```
Row 1: (0 * -0.5) + (1 * -0.5) + (0 * -0.5) = -0.5
Row 2: (0 *  2.0) + (1 *  2.0) + (0 *  2.0) =  2.0
Row 3: (0 * -0.5) + (1 * -0.5) + (0 * -0.5) = -0.5
Final score: -0.5 + 2.0 + -0.5 = 1.0
```

Low, as expected. The image has no horizontal bar.

### Biases and ReLU

A **bias** is another learned number, like a weight. It is added to the final score, and
you can think of it as a threshold the score has to clear.

```
                       vertical    horizontal
final score               6.0          1.0
bias                     -1.0         -2.0
output                    5.0         -1.0
```

This is where **ReLU** comes in. It replaces any negative output with zero:

```
                       vertical    horizontal
after ReLU                5.0          0.0
```

Layer 1 is done.

### Layer 2

Layer 1 took raw pixels and turned them into feature scores:

```
layer 1 output: [ vertical line: 5.0,  horizontal line: 0.0 ]
```

Layer 2 takes those feature scores as its input. Its weights do not look at pixels any
more; they look for *combinations of features* in order to make a final decision.

Say layer 2 is deciding whether the image is a 1 or a 7.

A 1 needs a strong vertical line and is penalized by a horizontal bar on top:

```
weights for "1": [ 2.0, -1.0 ]
                   ^    ^
                   |    how much I care about horizontal
                   how much I care about vertical

vertical:    5.0 * 2.0  = 10.0
horizontal:  0.0 * -1.0 =  0.0
final score: 10.0 + 0.0 = 10.0
```

A 7 needs both a horizontal top bar and a vertical stem:

```
weights for "7": [ 1.5, 2.0 ]

vertical:    5.0 * 1.5 = 7.5
horizontal:  0.0 * 2.0 = 0.0
final score: 7.5 + 0.0 = 7.5
```

Layer 2 has biases too, adjusting the standard each whole digit has to meet:

```
digit "1": bias  0.0 -> 10.0 +  0.0  = 10.0
digit "7": bias -3.0 ->  7.5 + (-3.0) = 4.5
```

### Argmax

```
layer 2 scores: [ digit 1: 10.0,  digit 7: 4.5 ]
```

Argmax looks at the final scores, sees that 10.0 is the largest, and returns its
position: 1.

### The point of layer 2

Layer 1 only answers "is there an edge or a line here?". Layer 2 answers "do these edges
add up to a whole shape?".

Without layer 2, the model could only check whether a pixel pattern matches a static
template. With layer 2 it can reason: IF vertical line is high AND horizontal line is
low, THEN it is a 1.

### The same thing, at the size this project uses

```
784 pixels (input)   raw intensity values of the 28x28 grid
layer 1 (784 -> 128) tests the 784 pixels against 128 basic feature filters — vertical
                     lines, horizontal bars, curves. The ReLU step, max(y, 0), zeros
                     out the non-matches, leaving 128 clean feature scores
layer 2 (128 -> 10)  evaluates those 128 feature scores against a pattern for each digit
                     0-9, giving 10 whole-digit scores
argmax               whichever of the 10 scores is largest is the answer
```

## Read next

[02-instruction-set.md](02-instruction-set.md).
