# Conference Room Rating Rubric

Used to produce `data/labels.csv`. Rate each photo as it is shown. The question is:
**"How ready is this room for the next meeting, as it stands?"**

All scores are integers from 1 to 5, except `overall_score`, which uses half points. 5 is best.

## Dimensions

### `clutter_score`: table and surfaces
| Score | Anchor |
|---|---|
| 5 | Table clear, or only intentional setup: place settings (glasses, notepads, pens laid out evenly), centrepiece, conference phone, name plates, built-in AV |
| 4 | One or two stray items (a cup, a bag on a side cabinet, a few papers) |
| 3 | Several leftover items on part of the table (cups, bottles, papers, laptops) |
| 2 | Items spread over most of the table |
| 1 | Table heavily covered or littered with trash or food |

### `chair_alignment_score`: chair arrangement
| Score | Anchor |
|---|---|
| 5 | Every chair tucked in or evenly placed, facing the table or front, matching rows |
| 4 | One or two chairs slightly out of line |
| 3 | Several chairs pulled out or rotated |
| 2 | Most chairs disordered |
| 1 | Chaotic: chairs scattered, stacked or knocked over |

Leave the field blank if no chairs are visible.

### `cleanliness_score`: floor, walls and general condition
| Score | Anchor |
|---|---|
| 5 | Floor clear, no visible cables, debris, boxes or stains; whiteboard clean or neatly blank |
| 4 | Minor issues: a loose cable, a box in a corner, faint whiteboard writing |
| 3 | Noticeable issues: items on the floor, messy cables, a written-on whiteboard |
| 2 | Several of the above |
| 1 | Dirty or damaged room, or rubbish on the floor |

### `overall_score`: holistic readiness
- Judge the room as a whole; this is not an average of the other scores.
- One serious problem, such as a littered table, can dominate the score.

## Rules
1. **Ignore decor, luxury, room size, lighting, image resolution and photo quality.** A plain, tidy room scores as high as a lavish, tidy one.
2. **Place settings are not clutter.** Evenly laid out glasses, water jugs, notepads or pens mean the room is prepared. Leftover cups, papers and bottles at random places are clutter.
3. **Occupied rooms** (`occupied = 1`): items in use (laptops, papers, cups) count as clutter, because they would still be there after the meeting. Judge chair alignment from the empty chairs and the overall arrangement.
4. **Rate only what is visible.** Use `confidence` (`high` / `medium` / `low`) for small, dark, partial or ambiguous photos.
5. **Not a conference room** (`valid_room = 0`): for an auditorium, classroom, restaurant, lobby and so on, still rate it, but flag it. These are excluded from training.

## Output columns
`filename, cleanliness_score, chair_alignment_score, clutter_score, overall_score, occupied, valid_room, confidence, notes`
