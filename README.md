# anagram-animator

This is a cleaned-up version of a Python script I've been using internally to render animated videos of anagrams. This is my first project with Pygame (even though it's not technically a game).

## Input format

Anagrams will be read from a file called `anagrams.txt` in the same directory as the Python script. Anagrams are expected to be separated by an equals sign (`=`). Punctuation and numbers are removed, and capitalization doesn't matter.

```
This is only an example = Oh, explains main style!
```

Multiple anagrams can be specified with multiple equals signs.

```
Multiple anagrams of a phrase
=
Is part of a large human sample
=
Animate some full paragraphs
```

Placing a linebreak in the middle of a phrase will force a linebreak to appear in that place in the rendered anagram.

```
Linebreaks are not only allowed,
but sometimes,
they're encouraged
=
The well made
enormous beauty
broken into large layered sections
```

Multiple sets of anagrams can be specified with single empty lines in between them.

```
first one = fine sort =
of insert = I sent for

The second one
= Oh, do sentence!

The third anagram on the list = I am short,
and then real tight
```

## Usage guide

Once the GUI starts up, there will be a section on the left with letter tiles, and a section on the right with various options and information. Pressing UP and DOWN will cycle through the options, and pressing LEFT and RIGHT will change their value.

| Option name                | Meaning                                                           |
|----------------------------|-------------------------------------------------------------------|
| Anagram                    | The index of the current anagram set.                             |
| Tiles per line             | Number of characters (including spaces) allowed in a single line. |
| Padding                    | Padding on sides, in tile space gaps.                             |
| Animation length (seconds) | Length of animation.                                              |
| Aspect ratio               | Aspect ratio of animation (any of 1:1, 9:16, or 16:9).            |

Pressing SPACE will preview the animation to the next phrase in the current anagram set, and pressing ENTER will render the anagram as a series of videos with keyframes in between them. The videos will be in a folder called `videos`, and the keyframes will be in a folder called `keyframes`. (Each animation frame will also be temporarily stored in a folder called `frames`.)

| Property name | Meaning                                             |
|---------------|-----------------------------------------------------|
| Rendered      | Whether a video for this anagram has been rendered. |
| Rendering     | Whether a video for this anagram is being rendered. |
| FPS           | Current FPS of the application.                     |
