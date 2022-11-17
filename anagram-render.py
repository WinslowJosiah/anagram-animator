import pygame
from pygame.color import Color
from pygame.font import Font
from pygame.locals import (
    K_LEFT, K_RIGHT, K_UP, K_DOWN,
    K_SPACE, K_RETURN,
    KEYDOWN, KEYUP,
    Rect, SRCALPHA, QUIT,
)
from pygame.sprite import Sprite
from pygame.surface import Surface
from pygame.time import Clock

from collections import Counter
from dataclasses import dataclass
from difflib import Match, SequenceMatcher
import hashlib
import numpy as np
import numpy.typing as npt
import os
import shutil
import textwrap
from typing import (
    Any, Callable, Generator, get_args, Literal, NamedTuple, Optional, Sequence
)


# Text alignment
Alignment = Literal["L", "R", "C"]


# Modes for rendering video
RenderModeID = Literal["kf_vid", "vid", "vid_loop"]
KEYFRAME_VIDEO, SINGLE_VIDEO, LOOPED_VIDEO = get_args(RenderModeID)


# GUI value with label
@dataclass
class LabelledValue:
    id: str
    label: str
    value: Any | Callable[..., Any]


# Set of dimensions, for different aspect ratios
class Dimension(NamedTuple):
    name: Optional[str]
    width: int
    height: int


# Rendering mode
class RenderMode(NamedTuple):
    label: str
    id: RenderModeID


# Type hints from Pygame (I could not figure out how to import them)
_RgbaOutput = tuple[int, int, int, int]
_ColorValue = (
    Color | int | str | tuple[int, int, int] | list[int] | _RgbaOutput
)


ROOT_DIR = os.path.realpath(os.path.dirname(__file__))


BLACK = (0, 0, 0)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
TAN = (236, 231, 201)
BROWN = (110, 69, 47)
DARK_GREEN = (32, 40, 32)


LINE_SPACING = -2


def normalize_phrase(s: str) -> str:
    """
    Modify a string into a normalized version with capital letters,
    spaces, and newlines.

    Parameters
    ----------
    s : str
        String to normalize.

    Returns
    -------
    str
        Normalized string.
    """
    return "\n".join(
        " ".join(
            "".join(
                c.upper()
                for c in word if c.isalpha()
            )
            for word in line.split()
        )
        for line in s.strip().split("\n")
    )


def is_anagram(x: str, y: str) -> bool:
    """
    Check whether two strings are anagrams.

    Parameters
    ----------
    x : str
        First phrase.
    y : str
        Second phrase.

    Returns
    -------
    bool
        True if `x` and `y` are anagrams.
    """
    return Counter(
        c.upper() for c in x if c.isalpha()
    ) == Counter(
        c.upper() for c in y if c.isalpha()
    )


# Algorithm from Chrobak et al. (2004)
# https://kam.mff.cuni.cz/~kolman/papers/mcsp.pdf
# There is an algorithm from He (2007) that is reportedly faster, but
# the speed difference isn't too noticeable from limited testing
# https://link.springer.com/chapter/10.1007/978-3-540-72031-7_40
def partition_anagram(s: str, t: str) -> tuple[list[str], list[str]]:
    """
    Find a ("near") minimum common substring partition of two anagrams.

    The Minimum Common Substring Partition problem (MCSP) is the problem
    of finding a partition of two strings which are anagrams, such that
    the parts of one can be reordered into the parts of the other. This
    problem is known to be NP-hard to solve optimally. A greedy
    algorithm is used here instead which runs in polynomial time.

    Note that while the greedy algorithm does not necessarily find the
    best solution, and has a bad worst-case approximation factor, in
    practice it finds decent solutions for English anagrams.

    Parameters
    ----------
    s : str
        Source string.
    t : str
        Target string.

    Returns
    -------
    tuple of (list of str, list of str)
        Two lists of strings which give the original strings when
        joined, and which can be rearranged to match each other.
    """
    assert is_anagram(s, t)

    # Both strings will use only uppercase letters and newlines
    s = "".join(c.upper() for c in s if c.isalpha() or c == "\n")
    t = "".join(c.upper() for c in t if c.isalpha() or c == "\n")

    # Newlines serve as pre-existing markers to avoid overlap between lines
    s_unmarked = set(i for i, c in enumerate(s) if c != "\n")
    t_unmarked = set(i for i, c in enumerate(t) if c != "\n")

    # Partition tables map indices to substrings
    s_partition: dict[int, str] = {}
    t_partition: dict[int, str] = {}

    # Helper function to get unmarked substrings and their indices
    def get_unmarked_substrings(
        x: str,
        x_unmarked: set[int]
    ) -> dict[int, str]:
        x_strings: dict[int, str] = {}

        partial = ""
        index = 0
        for i, c in enumerate(x):
            if i in x_unmarked:
                partial += c
            else:
                if partial:
                    x_strings[index] = partial
                partial = ""
                index = i + 1
        if partial:
            x_strings[index] = partial

        return x_strings

    # While there are unmarked symbols in S or T
    while s_unmarked or t_unmarked:
        # Get all unmarked substrings from S and T
        s_strings = get_unmarked_substrings(s, s_unmarked)
        t_strings = get_unmarked_substrings(t, t_unmarked)

        # Find the longest common substring over all pairs of substrings
        lcs = Match(0, 0, 0)
        lcs_string = ""
        for i, ss in s_strings.items():
            for j, st in t_strings.items():
                # Skip if both strings are shorter than LCS
                if len(ss) < lcs.size or len(st) < lcs.size:
                    continue

                match = SequenceMatcher(None, ss, st).find_longest_match()
                if match.size > lcs.size:
                    lcs_string = ss[match.a:match.a + match.size]
                    lcs = Match(i + match.a, j + match.b, match.size)

        # Add substring to partitions
        s_partition[lcs.a] = lcs_string
        t_partition[lcs.b] = lcs_string

        # Mark symbols of substring in S and T
        for i in range(lcs.size):
            s_unmarked.remove(lcs.a + i)
            t_unmarked.remove(lcs.b + i)

    # Return partitions as a tuple of lists of strings
    return (
        list(dict(sorted(s_partition.items())).values()),
        list(dict(sorted(t_partition.items())).values())
    )


def create_font(fonts: str | Sequence[str], *args: Any, **kwargs: Any) -> Font:
    """
    Create a `pygame.font.Font` object, choosing the first available
    font from a list of fonts.

    Parameters
    ----------
    fonts : str or array-like of str
        List of fonts (or a single font) to try creating.
    *args
        These parameters will be passed to the `pygame.font.Font`
        constructor.
    **kwargs
        These parameters will be passed to the `pygame.font.Font`
        constructor.
    """
    if isinstance(fonts, str):
        fonts = [fonts]

    available = pygame.font.get_fonts()
    choices = map(lambda s: s.lower().replace(" ", ""), fonts)
    for choice in choices:
        if choice in available:
            return pygame.font.SysFont(choice, *args, **kwargs)
    return pygame.font.Font(None, *args, **kwargs)


def word_wrap_text(
    surface: Surface,
    text: str,
    color: _ColorValue,
    rect: Rect | tuple[float, float, float, float] | list[float],
    font: Font,
    anti_alias: bool = False,
    bg_color: Optional[_ColorValue] = None,
    align: Alignment = "L"
) -> tuple[str, int]:
    """
    Render some text onto a certain area of a surface, wrapping words
    automatically.

    Parameters
    ----------
    surface : pygame.Surface
        Surface to render text onto.
    text : str
        Text to render.
    color : pygame.color.Color
        Color to render text with.
    rect : pygame.rect.Rect
        Area to render text in on `surface`.
    font : pygame.font.Font
        Font to render text with.
    anti_alias : bool, default False
        If true, apply anti-aliasing.
    bg_color : pygame.color.Color, optional
        Background color to render text with.
    align : str, default "L"
        Alignment of text (any of "L", "R", or "C").

    Returns
    -------
    tuple of (str, int)
        Text that was not rendered, and Y-value of the line after the
        last.
    """
    _rect = Rect(rect)
    y = _rect.top

    font_height = font.size("Tg")[1]

    while text:
        i = 1

        # If this row is outside our area, quit
        if y + font_height > _rect.bottom:
            break

        # Determine maximum width of line
        while font.size(text[:i])[0] < _rect.width and i < len(text):
            i += 1

        j = i
        # If we're not past the end of the line
        if i < len(text):
            # Find end of last word
            j = text.rfind(" ", 0, i)
            # Adjust wrap to end of last word if found
            if j > 0:
                i = j
            else:
                j = i

        # Render the line
        if bg_color:
            image = font.render(text[:i], True, color, bg_color)
            image.set_colorkey(bg_color)
        else:
            image = font.render(text[:i], anti_alias, color)

        match align:
            case "L":  # left-align
                x = _rect.left
            case "R":  # right-align
                x = _rect.right - image.get_width()
            case "C":  # center-align
                x = _rect.centerx - image.get_width() // 2
            case _:  # just in case
                x = _rect.left

        # Blit the line to the surface
        surface.blit(image, (x, y))
        y += font_height + LINE_SPACING

        text = text[j:]

    return text, y


class Tile(Sprite):
    LETTER_HEIGHT_TO_TILE_SIDE = 7.2 / 12.5
    # *4 to make the border radius more visible; remove for realistic value
    BORDER_RADIUS_TO_TILE_SIDE = .3 / 12.5 * 4
    LETTER_HEIGHT_TO_FONT_SIZE = 16 / 12
    FONT_SIZE_TO_TILE_SIDE = (
        LETTER_HEIGHT_TO_FONT_SIZE * LETTER_HEIGHT_TO_TILE_SIDE
    )

    ANIM_SMOOTHNESS = 2.75
    Z_FACTOR = 1 / 3.75

    def __init__(
        self,
        letter: str,
        side: int | float,
        x: int | float,
        y: int | float,
    ):
        """
        Create a Bananagrams-style letter tile sprite.

        Parameters
        ----------
        letter : str
            Letter displayed on the tile.
        side : int or float
            Side length of the tile in pixels.
        x : int or float
            X position of the tile.
        y : int or float
            Y position of the tile.
        """
        super(Tile, self).__init__()

        assert letter
        assert letter[0].isalpha()
        self.letter = letter[0].upper()
        self.side = int(side)
        self.font = create_font(
            "Arial",
            int(self.side * self.FONT_SIZE_TO_TILE_SIDE)
        )

        # Image for rendering the tile
        self.render_image = Surface((self.side, self.side), flags=SRCALPHA)
        # Image used for blitting the tile to the screen
        self.image = Surface((self.side, self.side), flags=SRCALPHA)
        self.rect: Rect = self.image.get_rect()

        # Source position
        self.pos: tuple[int | float, int | float] = (x, y)
        # Destination position
        self.dest: tuple[int | float, int | float] = (x, y)
        # Is the tile following a path to the destination?
        self.following_path = False
        # Progress along path, from 0 to 1
        self.path_t: float = 0
        # Path progress delta
        self.path_dt = 0.05
        # Is the tile moving up or down?
        self.going_up = False
        # "Height" of the tile (used to scale image)
        self.z: float = 0

        self.rect.center = (int(x), int(y))
        self.draw()

    def update(self, *args: Any, **kwargs: Any):
        x: int | float
        y: int | float
        x, y = self.pos

        # If this tile is following a path
        if self.following_path:
            dx, dy = self.dest

            # If path progress is non-negative
            if self.path_t >= 0:
                # This tile will follow some cubic Bezier curve
                # with control points related to its source and destination

                # Calculate how much the path should arc upwards
                # (the greater the X distance, the greater the arc)
                pdy = abs(x - dx) / 1.5
                # If the arc height is greater than the Y distance
                if pdy > abs(y - dy):
                    max_pdy = 3 * self.side
                    # Cap arc height to maximum allowed arc height,
                    # but only if it's still greater than the Y distance
                    if pdy > max_pdy > abs(y - dy):
                        pdy = max_pdy

                    # Arc goes down by default; switch direction if necessary
                    if self.going_up:
                        pdy *= -1

                    # The Y inflection point is the Y plus the arc height
                    py = y + pdy
                # If the Y distance is not less than the arc height
                else:
                    # The Y inflection point is halfway between the Y values
                    py = (y + dy) / 2

                # Create control points of Bezier curve
                P0: npt.NDArray[np.float32] = np.array(  # type: ignore
                    [x, y], dtype=np.float32
                )
                P1: npt.NDArray[np.float32] = np.array(  # type: ignore
                    [x, py], dtype=np.float32
                )
                P2: npt.NDArray[np.float32] = np.array(  # type: ignore
                    [dx, py], dtype=np.float32
                )
                P3: npt.NDArray[np.float32] = np.array(  # type: ignore
                    [dx, dy], dtype=np.float32
                )

                # Ease T value with polynomial curve (this works, trust me)
                t = (
                    ((2 * self.path_t) ** self.ANIM_SMOOTHNESS) / 2
                    if self.path_t < .5 else
                    1 - ((2 - 2 * self.path_t) ** self.ANIM_SMOOTHNESS) / 2
                )
                mt = 1 - t

                # Calculate point along Bezier curve using the polynomial form
                vector: npt.NDArray[np.float32] = (
                    mt ** 3 * P0
                    + 3 * t * mt ** 2 * P1
                    + 3 * t ** 2 * mt * P2
                    + t ** 3 * P3
                )
                x, y = vector

                # Set Z based on distance between our point and the destination
                origin_vector: npt.NDArray[np.float32] = vector - P3
                dist = np.sqrt(
                    origin_vector.dot(origin_vector)  # type: ignore
                ) / self.side
                self.z = t * mt * dist * self.Z_FACTOR

            # Step path progress forward
            self.path_t += self.path_dt
            # If path is done
            if self.path_t >= 1:
                # Set position to the destination
                x, y = dx, dy
                self.pos = (x, y)
                # Reset path variables
                self.z = 0
                self.following_path = False
                self.path_t = 0

            # Re-render sprite
            self.draw()

        # Set position of sprite
        self.rect.center = (int(x), int(y))

    def draw(self):
        # Fill with transparency
        self.render_image.fill((0, 0, 0, 0))

        # Draw rectangle with tile color
        pygame.draw.rect(
            self.render_image,
            TAN,
            (0, 0, self.side, self.side),
            border_radius=int(self.side * self.BORDER_RADIUS_TO_TILE_SIDE),
        )

        # Draw border around tile
        pygame.draw.rect(
            self.render_image,
            BLACK,
            (0, 0, self.side, self.side),
            width=int(self.side * self.BORDER_RADIUS_TO_TILE_SIDE / 1.25),
            border_radius=int(self.side * self.BORDER_RADIUS_TO_TILE_SIDE),
        )

        # Render letter
        letter_surface = self.font.render(self.letter, True, BLACK)
        letter_w, letter_h = letter_surface.get_size()
        # Stretch letter horizontally
        letter_surface = pygame.transform.smoothscale(
            letter_surface,
            (letter_w * 1.25, letter_h)
        )
        # Blit letter to center of image
        letter_rect = letter_surface.get_rect()
        letter_rect.center = (self.side // 2, self.side // 2)
        self.render_image.blit(letter_surface, letter_rect)
        self.image = self.render_image

        # If tile is following path, scale it by Z
        if self.following_path:
            if self.path_t >= 0:
                self.image = pygame.transform.smoothscale(
                    self.render_image,
                    (self.side * (1 + self.z), self.side * (1 + self.z)),
                )


class Game:
    FPS = 30

    TILE_GAP_TO_TILE_SIDE = 1 / 15
    TILE_SPACE_TO_TILE_SIDE = 1 / 2

    OPTIONS_WIDTH_PROPORTION = 1.0
    LABELS_WIDTH_PROPORTION = 0.6

    DIMENSIONS = [
        Dimension("1:1", 1080, 1080),
        Dimension("9:16", 1080, 1920),
        Dimension("16:9", 1920, 1080),
    ]

    RENDER_NAME_FORMATS: dict[RenderModeID, str] = {
        KEYFRAME_VIDEO: "{hash}_{kf}",
        SINGLE_VIDEO: "{hash}_full",
        LOOPED_VIDEO: "{hash}_loop",
    }

    def __init__(self, anagrams: list[list[str]]):
        """
        Create a GUI for an anagram animation renderer.

        Parameters
        ----------
        anagrams : list of list of str
            List of anagrams to load into renderer, each of which are
            represented as lists of phrases which are anagrams.
        """
        pygame.init()

        self.RENDER_MODES = [
            RenderMode("Keyframes + videos", KEYFRAME_VIDEO),
            RenderMode("Single video (start to end)", SINGLE_VIDEO),
            RenderMode("Single video (loop)", LOOPED_VIDEO),
        ]

        # Listed options and properties
        # (this isn't really the best way to do this)
        self.OPTIONS: list[LabelledValue] = [
            LabelledValue("ana", "Anagram", lambda: self.anagrams_i + 1),
            LabelledValue("tpl", "Tiles per line", lambda: self.line_width),
            LabelledValue("pad", "Padding", lambda: self.side_padding),
            LabelledValue("len", "Animation length (seconds)", lambda: (
                self.anim_length
            )),
            LabelledValue("pau", "Animation pause (seconds)", lambda: (
                self.anim_pause
            )),
            LabelledValue("rat", "Aspect ratio", lambda: (
                self.DIMENSIONS[self.dimension_i].name
            )),
            LabelledValue("mod", "Render mode", lambda: (
                self.RENDER_MODES[self.render_mode_i].label
            )),
        ]
        self.PROPERTIES: list[LabelledValue] = [
            LabelledValue("rdd", "Rendered", lambda: (
                "Yes"
                if any(
                    os.path.isfile(os.path.join(
                        self.VIDEOS, fmt.format(
                            hash=self.get_phrases_hash(),
                            kf=0
                        ) + ".mp4"
                    ))
                    for fmt in self.RENDER_NAME_FORMATS.values()
                )
                else "No"
            )),
            LabelledValue("rdn", "Rendering", lambda: (
                "Yes" if self.rendering else "No"
            )),
            LabelledValue("fps", "FPS", lambda: round(self.clock.get_fps(), 3)),
        ]

        # List of anagrams passed into game
        self.anagrams: list[list[str]] = anagrams
        self.texts_i: int = 0

        # Option values
        self.anagrams_i: int = 0
        self.option_i: int = 0
        self.dimension_i: int = 0
        self.render_mode_i: int = 0
        self.anim_length: float = 2.0
        self.anim_pause: float = 3.0
        self.side_padding: int = 4

        # Set initial window dimensions
        self.set_window_dimensions()

        # Load font
        self.font_size: int = int(
            min(self.WINDOW_WIDTH, self.WINDOW_HEIGHT) / 27
        )
        self.font: Font = create_font(
            ("Segoe UI", "Tahoma", "Lucida Grande", "Arial"),
            self.font_size,
            bold=True
        )

        # Clock for regulating FPS
        self.clock: Clock = pygame.time.Clock()

        # Title of game window
        pygame.display.set_caption("Anagram Animator")

        # Directories
        self.FRAMES = os.path.join(ROOT_DIR, "frames")
        self.KEYFRAMES = os.path.join(ROOT_DIR, "keyframes")
        self.VIDEOS = os.path.join(ROOT_DIR, "videos")

        # Keyboard states
        self.keys_down: list[int] = []
        self.keys_up: list[int] = []
        self.keys_pressed: list[int] = []

    def run(self) -> bool:
        self.init()

        # Main loop
        while True:
            # Update frame and draw
            should_quit = self.update()
            self.draw()
            pygame.display.update()

            # Tick one frame
            if not self.rendering:
                self.clock.tick(self.FPS)

            if should_quit:
                pygame.quit()
                return False

            # Set up keyboard state lists
            prev_keys_down: list[int] = self.keys_down.copy()
            self.keys_down.clear()
            self.keys_up.clear()
            self.keys_pressed.clear()

            # Process Pygame events
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return True

                if event.type == KEYDOWN:
                    self.keys_down.append(event.key)
                    if event.key not in prev_keys_down:
                        self.keys_pressed.append(event.key)

                if event.type == KEYUP:
                    self.keys_up.append(event.key)

    def init(self):
        # Set variables to be reset upon init
        self.line_width = 0  # temp value; this is corrected

        # Set up surfaces
        self.render_surface = Surface(
            (self.RENDER_WIDTH, self.RENDER_HEIGHT), flags=SRCALPHA
        )
        self.options_surface: Surface = Surface(
            (self.OPTIONS_WIDTH, self.OPTIONS_HEIGHT), flags=SRCALPHA
        )

        # Set up animation
        self.setup_animation()
        self.rendering = False
        self.animating = False

        # Create tiles
        self.texts_i = 0
        self.tiles: list[Tile] = self.phrase_to_tiles(
            self.get_nth_phrase(self.texts_i)
        )

        # Set up pseudo-coroutine
        self.update_iter: Optional[Generator[Any, None, None]] = None

        # Clear keyboard state
        self.keys_down.clear()
        self.keys_up.clear()
        self.keys_pressed.clear()

    def update(self) -> bool:
        # If pseudo-coroutine is active, try stepping it forward
        should_return: bool = False
        if self.update_iter is not None:
            try:
                should_return = next(self.update_iter)
            except StopIteration:
                self.update_iter = None
            return should_return

        # Check whether LEFT and RIGHT were pressed
        left_pressed = K_LEFT in self.keys_pressed
        right_pressed = K_RIGHT in self.keys_pressed

        should_setup = False
        should_create_tiles = False
        should_init = False
        should_set_dims = False
        # Handle updating for the current option (based on option ID)
        match self.OPTIONS[self.option_i].id:
            case "ana":  # Anagram
                if left_pressed:
                    self.anagrams_i -= 1
                    should_init = True
                if right_pressed:
                    self.anagrams_i += 1
                    should_init = True
                self.anagrams_i %= len(self.anagrams)
            case "tpl":  # Tiles per line
                if left_pressed:
                    if self.line_width > self.MIN_LINE_WIDTH:
                        self.line_width -= 1
                        should_setup = True
                        should_create_tiles = True
                if right_pressed:
                    if self.line_width < self.MAX_LINE_WIDTH:
                        self.line_width += 1
                        should_setup = True
                        should_create_tiles = True
            case "pad":  # Padding
                if left_pressed:
                    if self.side_padding > 0:
                        self.side_padding -= 1
                        should_setup = True
                        should_create_tiles = True
                if right_pressed:
                    if self.side_padding < 16:
                        self.side_padding += 1
                        should_setup = True
                        should_create_tiles = True
            case "len":  # Animation length (seconds)
                if left_pressed:
                    if self.anim_length > 1:
                        self.anim_length -= 0.125
                if right_pressed:
                    if self.anim_length < 3:
                        self.anim_length += 0.125
            case "pau":  # Animation pause (seconds)
                if left_pressed:
                    if self.anim_pause > 1:
                        self.anim_pause -= 0.125
                if right_pressed:
                    if self.anim_pause < 10:
                        self.anim_pause += 0.125
            case "rat":  # Aspect ratio
                if left_pressed:
                    self.dimension_i -= 1
                    should_init = True
                    should_set_dims = True
                if right_pressed:
                    self.dimension_i += 1
                    should_init = True
                    should_set_dims = True
                self.dimension_i %= len(self.DIMENSIONS)
            case "mod":  # Render mode
                if left_pressed:
                    self.render_mode_i -= 1
                if right_pressed:
                    self.render_mode_i += 1
                self.render_mode_i %= len(self.RENDER_MODES)
            case _:  # just in case
                pass

        # Call functions that need to be called after setting options
        if should_set_dims:
            self.set_window_dimensions()
        if should_setup:
            self.setup_animation()
        if should_create_tiles:
            self.tiles = self.phrase_to_tiles(
                self.get_nth_phrase(self.texts_i)
            )
        if should_init:
            self.init()

        # UP and DOWN to change which option is selected
        if K_UP in self.keys_pressed:
            self.option_i = max(self.option_i - 1, 0)
        if K_DOWN in self.keys_pressed:
            self.option_i = min(self.option_i + 1, len(self.OPTIONS) - 1)

        # SPACE to animate to next phrase
        if K_SPACE in self.keys_pressed:
            self.update_iter = iter(self.functional_animate())

        # RETURN to render
        if K_RETURN in self.keys_pressed:
            self.update_iter = iter(self.functional_render(
                self.RENDER_MODES[self.render_mode_i].id
            ))

        return should_return

    def render_tiles(self):
        """
        Render tiles onto rendering surface.
        """
        self.render_surface.fill(GREEN)
        # Blit Z-sorted tiles
        for tile in sorted(self.tiles, key=lambda s: s.z):
            assert tile.image is not None
            assert tile.rect is not None
            self.render_surface.blit(tile.image, tile.rect)

    def draw(self):
        # Render to render surface
        if not self.rendering:
            self.render_tiles()

        # Render to options surface

        # Fill background
        self.options_surface.fill(DARK_GREEN)

        # Set some useful constants
        padding = int(self.font_size / 2)
        font_height = self.font.size("Tg")[1]

        # Set some more useful constants
        full_width = self.OPTIONS_WIDTH - padding * 2
        labels_plus_values_width = full_width - padding
        labels_width = int(
            labels_plus_values_width * self.LABELS_WIDTH_PROPORTION
        )
        values_width = int(labels_plus_values_width - labels_width)
        max_label_height = (font_height + LINE_SPACING) * 3 - LINE_SPACING

        x, y = padding, padding
        # Loop through each option
        for i, option in enumerate(self.OPTIONS):
            # Display the option's label
            _, lty = word_wrap_text(
                self.options_surface,
                option.label,
                WHITE if i == self.option_i else GRAY,
                (x, y, labels_width, max_label_height),
                self.font, True, align="R"
            )

            # Convert option's value to string
            # (calling it as a function if necessary)
            if isinstance(option.value, Callable):
                value_str = str(option.value())
            else:
                value_str = str(option.value)

            # Display the option's value
            _, vty = word_wrap_text(
                self.options_surface,
                value_str,
                WHITE,
                (
                    x + labels_width + padding, y,
                    values_width, max_label_height
                ),
                self.font, True, align="C"
            )

            # x, y = x, ty
            x, y = x, max(lty, vty)

        x, y = x, y + font_height + LINE_SPACING
        # Loop through each property
        for prop in self.PROPERTIES:
            # Display the property's label
            _, ty = word_wrap_text(
                self.options_surface,
                prop.label,
                WHITE,
                (x, y, labels_width, max_label_height),
                self.font, True, align="R"
            )

            # Convert property's value to string
            # (calling it as a function if necessary)
            if isinstance(prop.value, Callable):
                value_str = str(prop.value())
            else:
                value_str = str(prop.value)

            # Display the property's value
            word_wrap_text(
                self.options_surface,
                value_str,
                WHITE,
                (
                    x + labels_width + padding, y,
                    values_width, ty - y - LINE_SPACING
                ),
                self.font, True, align="C"
            )

            x, y = x, ty

        # Render some extra instructional text
        x, y = x, y + font_height + LINE_SPACING
        for line in (
            "UP/DOWN to go through options",
            "LEFT/RIGHT to change value",
            "SPACE to preview animation",
            "ENTER to render",
        ):
            _, ty = word_wrap_text(
                self.options_surface,
                line,
                WHITE, (x, y, full_width, max_label_height),
                self.font, True
            )
            x, y = x, ty

        # Render to window

        # Blit scaled version of rendering surface
        self.surface.blit(
            pygame.transform.smoothscale(
                self.render_surface, (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
            ),
            (0, 0)
        )

        # Blit options surface
        self.surface.blit(self.options_surface, (self.PREVIEW_WIDTH, 0))

    def functional_animate(self) -> Generator[Any, None, None]:
        # Set up animation
        self.setup_animation()
        self.animating = True

        # Create tiles for this phrase
        self.tiles = self.phrase_to_tiles(
            self.get_nth_phrase(self.texts_i)
        )

        # Animate the tiles to their destinations
        self.set_tile_anims(self.anim_length, self.texts_i)
        while self.animating:
            for tile in self.tiles:
                tile.update()

            yield False

            last_tile = self.tiles[-1]
            self.animating = last_tile.following_path

        # Advance to the next phrase
        self.texts_i = (self.texts_i + 1) % len(self.get_phraselist())

    def functional_render(self, mode: RenderModeID) -> Generator[Any, None, None]:
        # Create hash from phrases in anagram
        phrases_hash = self.get_phrases_hash()

        # Get name of rendered video file
        video_name_format = self.RENDER_NAME_FORMATS[mode]

        # Set up animation
        self.setup_animation()

        # Create videos (and maybe keyframes) directory
        if mode == KEYFRAME_VIDEO and not os.path.isdir(self.KEYFRAMES):
            os.mkdir(self.KEYFRAMES)
        if not os.path.isdir(self.VIDEOS):
            os.mkdir(self.VIDEOS)

        # Set up render
        self.setup_render()
        self.keyframe = 0
        self.rendering = True

        # Set the number of phrases to animate
        number_of_phrases = len(self.get_phraselist())
        if mode in (KEYFRAME_VIDEO, SINGLE_VIDEO):
            number_of_phrases -= 1

        # Main animation loop
        for i in range(number_of_phrases):
            # Create tiles for this phrase
            self.tiles = self.phrase_to_tiles(
                self.get_nth_phrase(i)
            )

            # If rendering keyframe video
            if mode == KEYFRAME_VIDEO:
                # Save the pre-animation keyframe
                self.render_tiles()
                self.save_keyframe(f"{phrases_hash}_{self.keyframe}")
                self.keyframe += 1
                yield False
            # If rendering single or looped video
            elif mode in (SINGLE_VIDEO, LOOPED_VIDEO):
                # Pause for the specified length of time
                self.render_tiles()
                for _ in range(int(self.anim_pause * self.FPS)):
                    self.save_frame()
                    self.frame += 1
                    yield False

            # Animate the tiles to their destinations
            self.set_tile_anims(self.anim_length, i)
            self.animating = True
            while self.animating:
                for tile in self.tiles:
                    tile.update()

                # Render and save frame
                self.render_tiles()
                self.save_frame()
                self.frame += 1
                yield False

                last_tile = self.tiles[-1]
                self.animating = last_tile.following_path

            # The tiles have been animated to their destinations
            # If rendering keyframe video
            if mode == KEYFRAME_VIDEO:
                # Render this video, and set up again
                self.render_video(video_name_format.format(
                    hash=phrases_hash,
                    kf=self.keyframe - 1,
                ))
                self.setup_render()

        # The main animation loop is done
        # If rendering single video
        if mode == SINGLE_VIDEO:
            # Pause for the specified length of time
            self.render_tiles()
            for _ in range(int(self.anim_pause * self.FPS)):
                self.save_frame()
                self.frame += 1
                yield False

        # The main animation loop has ended
        # If rendering keyframe video
        if mode == KEYFRAME_VIDEO:
            # Save the post-animation keyframe
            self.save_keyframe(f"{phrases_hash}_{self.keyframe}")
        # If rendering single or looped video
        elif mode in (SINGLE_VIDEO, LOOPED_VIDEO):
            # Render this video, and set up again
            self.render_video(video_name_format.format(
                hash=phrases_hash,
                kf=self.keyframe - 1,
            ))
            self.setup_render()

        # Transition out of render
        self.setup_animation()
        self.tiles = self.phrase_to_tiles(
            self.get_nth_phrase(self.texts_i)
        )
        self.rendering = False

    def set_window_dimensions(self):
        """
        Set dimensions of this window.
        """
        dimension = self.DIMENSIONS[self.dimension_i]
        self.RENDER_WIDTH = dimension.width
        self.RENDER_HEIGHT = dimension.height
        self.PREVIEW_HEIGHT = 480
        self.PREVIEW_WIDTH = (
            self.PREVIEW_HEIGHT * self.RENDER_WIDTH / self.RENDER_HEIGHT
        )
        self.OPTIONS_HEIGHT = self.PREVIEW_HEIGHT
        self.OPTIONS_WIDTH = min(
            self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT
        ) * self.OPTIONS_WIDTH_PROPORTION
        self.WINDOW_WIDTH = self.PREVIEW_WIDTH + self.OPTIONS_WIDTH
        self.WINDOW_HEIGHT = self.PREVIEW_HEIGHT

        # Recreate window
        pygame.display.quit()
        self.surface: Surface = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        )

    def get_phraselist(self) -> list[str]:
        """
        Get the current list of phrases in the list of anagram phrases.

        Returns
        -------
        list of str
            Current phrase list.
        """
        return self.anagrams[self.anagrams_i]

    def get_nth_phrase(self, n: int) -> str:
        """
        Get the `n`th phrase in the current list of phrases for this
        anagram, wrapping around when out of range.

        Parameters
        ----------
        n : int
            Index into phrase list.

        Returns
        -------
        str
            `n`th phrase.
        """
        texts = self.get_phraselist()
        return texts[n % len(texts)]

    def get_phrases_hash(self) -> str:
        """
        Get hash for current phrases.
        """
        return hashlib.shake_128(
            bytes("\n=\n".join(self.get_phraselist()), "utf-8")
        ).hexdigest(12)

    def setup_animation(self):
        """
        Set up variables before tile animation.
        """
        tile_width = 1 + self.TILE_GAP_TO_TILE_SIDE

        # Calculate how many letters (minimum) to fit on a line
        # At minimum, this will be the length of the longest word
        min_lw: int = max(
            max(len(word) for word in text.split())
            for text in self.get_phraselist()
        )

        # Try increasing the minimum line width if necessary
        while True:
            # Calculate tile side length for current width
            tile_side_for_w = self.RENDER_WIDTH / (
                min_lw * tile_width + self.side_padding
            )

            all_phrases_fit = True
            for text in self.get_phraselist():
                lh = len(self.phrase_to_lines(text, min_lw))

                # Calculate tile side length for current height
                tile_side_for_h = self.RENDER_HEIGHT / (
                    lh * tile_width + self.side_padding
                )

                # If tile side length cannot fit tiles within this height
                if tile_side_for_w > tile_side_for_h:
                    # This phrase doesn't fit for this tile side length
                    all_phrases_fit = False
                    break

            # If all phrases fit, stop increasing
            if all_phrases_fit:
                break

            # If not all phrases fit, increase by 1 and test again
            min_lw += 1

        # Calculate how many letters (maximum) to fit on a line
        # At maximum, this will be the length of a single line
        max_lw: int = max(
            max(len(line) for line in text.split("\n"))
            for text in self.get_phraselist()
        )

        # Set line width restrictions
        self.MIN_LINE_WIDTH: int = min_lw
        self.MAX_LINE_WIDTH: int = max_lw

        # Correct line width
        self.line_width = min(
            max(self.line_width, self.MIN_LINE_WIDTH),
            self.MAX_LINE_WIDTH
        )

        # Set tile side length
        self.TILE_SIDE: float = self.RENDER_WIDTH / (
            self.line_width * tile_width + self.side_padding
        )

    def setup_render(self):
        """
        Set up the render of this video.

        The frames directory will be created (emptied, if it exists),
        and the current frame will be set to 0.
        """
        if os.path.isdir(self.FRAMES):
            shutil.rmtree(self.FRAMES)
        os.mkdir(self.FRAMES)
        self.frame = 0

    def render_video(self, filename: str):
        """
        Render this video.

        Parameters
        ----------
        filename : str
            Filename (without extension).
        """
        os.system(
            f"ffmpeg -y -framerate {self.FPS} "
            f"-i \"{os.path.join(self.FRAMES, '%06d.png')}\" "
            f"-c:v libx264 -pix_fmt yuv420p "
            f"\"{os.path.join(self.VIDEOS, filename)}.mp4\""
        )

    def save_frame(self):
        """
        Save this frame.
        """
        pygame.image.save_extended(
            self.render_surface,
            os.path.join(self.FRAMES, f"{self.frame:06}.png")
        )

    def save_keyframe(self, filename: str):
        """
        Save this keyframe.

        Parameters
        ----------
        filename : str
            Filename without extension.
        """
        pygame.image.save_extended(
            self.render_surface,
            os.path.join(self.KEYFRAMES, f"{filename}.png")
        )

    def set_tile_anims(self, seconds: int | float, i: int):
        """
        Set the animation properties for each tile, so they animate
        from spelling phrase `i` to spelling phrase `i + 1` in `seconds`
        seconds.

        Parameters
        ----------
        seconds : int or float
            Duration of the animation in seconds.
        i : int
            Index of first phrase into phrase list.
        """
        # Create tiles for second phrase
        phrase_a = self.get_nth_phrase(i)
        phrase_b = self.get_nth_phrase(i + 1)
        self.tile_dests: list[Tile] = self.phrase_to_tiles(phrase_b)

        # Find a partition for the two phrases
        x_partition, y_partition = partition_anagram(
            "\n".join(self.phrase_to_lines(phrase_a)),
            "\n".join(self.phrase_to_lines(phrase_b))
        )

        # Create list of indices for second phrase partition
        y_indices: list[int] = []
        i = 0
        for s in y_partition:
            y_indices.append(i)
            i += len(s)

        going_up = True
        letter_i = 0
        total_letters = len([c for c in phrase_a if c.isalpha()])
        # Create iterator through letter tiles
        tiles_iter = iter(self.tiles)
        # For every string in first phrase partition
        for s in x_partition:
            # Find this string in second phrase partition
            j = y_partition.index(s)
            k = y_indices[j]

            switch = True
            # For every character in this substring
            for i in range(len(s)):
                # Get the next tile
                tile = next(tiles_iter)
                # Get the position of this tile and the corresponding
                # destination tile
                x, _ = tile.pos
                dx, dy = self.tile_dests[k + i].pos
                # Set destination position to position of destination tile
                tile.dest = (dx, dy)
                # Set animation speed to take a certain amount of seconds
                tile.path_dt = 1 / (self.FPS * seconds)
                # If going_up is true, the tile goes up
                tile.going_up = going_up
                # Set animation progress based on current letter index
                tile.path_t = -letter_i / (total_letters * 2.5)
                # Set this tile to start animating
                tile.following_path = True
                # Not sure why I added this, but...
                # If X distance is less than about 2.1 tiles + tile gaps
                if abs(x - dx) < 2.1 * self.TILE_SIDE * (
                    1 + self.TILE_GAP_TO_TILE_SIDE
                ):
                    # Do not switch whether the next tile goes up/down
                    # (Otherwise, do switch)
                    switch = False
            letter_i += len(s)

            # Delete this substring from the second phrase partition
            del y_indices[j]
            del y_partition[j]

            # Switch whether the next tile goes up if necessary
            if switch:
                going_up = not going_up

    def phrase_to_lines(
        self,
        phrase: str,
        line_width: Optional[int] = None
    ) -> list[str]:
        """
        Convert phrase to list of lines.

        Parameters
        ----------
        phrase : str
            Phrase to convert to lines.

        line_width : int, default `self.line_width`
            Maximum width of each line.

        Returns
        -------
        list of str
            List of lines created.
        """
        if line_width is None:
            line_width = self.line_width

        return "\n".join(
            "\n".join(
                l.strip() for l in textwrap.wrap(
                    line, line_width,
                    replace_whitespace=False,
                )
            )
            for line in phrase.splitlines()
        ).splitlines()

    def phrase_to_tiles(self, phrase: str) -> list[Tile]:
        """
        Convert phrase to list of Tile objects.

        Parameters
        ----------
        phrase : str
            Phrase to convert to tiles.

        Returns
        -------
        list of Tile
            List of Tiles created.
        """
        lines = self.phrase_to_lines(phrase)
        tiles: list[Tile] = []

        gap = self.TILE_SIDE * self.TILE_GAP_TO_TILE_SIDE
        space = self.TILE_SIDE * self.TILE_SPACE_TO_TILE_SIDE

        # Calculate height of lines and starting Y
        height = len(lines) * (self.TILE_SIDE + gap) - gap
        y = (self.RENDER_HEIGHT + self.TILE_SIDE - height) / 2

        for line in lines:
            # Calculate width of line and starting X
            width = sum(
                (space if c.isspace() else self.TILE_SIDE) + gap
                for c in line
            ) - gap
            x = (self.RENDER_WIDTH + self.TILE_SIDE - width) / 2

            for c in line:
                # If character is space, advance X by space
                if c.isspace():
                    x += space + gap
                    continue
                # If character is letter, add letter tile
                tiles.append(Tile(c.upper(), self.TILE_SIDE, x, y))
                # Advance X by tile side
                x += self.TILE_SIDE + gap
            # Advance Y by tile side
            y += self.TILE_SIDE + gap

        return tiles


def main():
    filename = os.path.join(ROOT_DIR, "anagrams.txt")

    try:
        with open(filename) as f:
            file_text: str = f.read()
    except FileNotFoundError:
        print(f"ERROR: {filename} does not exist!")
        return

    anagrams: list[list[str]] = []
    for i, entry in enumerate(file_text.split("\n\n")):
        print(f"Parsing anagram {i + 1}...")

        phrases = entry.split("=")
        if len(phrases) < 2:
            print("ERROR: Missing '=' symbol!")
            return

        phrases = [normalize_phrase(p) for p in phrases]

        if any(
            not is_anagram(phrases[i], phrases[i+1])
            for i in range(len(phrases) - 1)
        ):
            print("ERROR: Phrases aren't anagrams!")
            print("\n=/=\n".join(phrases))
            return

        anagrams.append(phrases)

    print("Anagrams parsed.")

    game = Game(anagrams)
    was_quit = game.run()

    if was_quit:
        print("User has quit.")
        return

if __name__ == "__main__":
    main()
