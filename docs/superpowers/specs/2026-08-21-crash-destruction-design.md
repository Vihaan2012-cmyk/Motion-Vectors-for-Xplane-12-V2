# Crash Destruction — Design

Visual structural destruction for the user's aircraft in X-Plane 12: the
airframe fragments into 800–2000 pieces on impact, pieces stay attached until
their joints break, and the wreckage settles on the ground instead of sinking
through it.

Flight dynamics remain X-Plane's. This is a rendering and debris system, not a
physics replacement.

## Why the obvious approach does not work

The natural design is to fragment the aircraft mesh: read its geometry, cut it
into chunks, simulate the chunks. The Vulkan layer cannot do that. It sees
draw calls and vertex buffers, not objects — an aircraft arrives as many draws
with X-Plane's own animation transforms applied to gear, flaps and control
surfaces. Reconstructing "this is a wing" from that is the whole problem, and
it would have to be redone for every add-on aircraft.

A second natural approach is to classify vertices by their **object-space**
position against a set of boxes. That fails for a subtler reason: every object
has its own local space centred near its origin, so a hangar's local space is
indistinguishable from the aircraft's. Classifying in object space requires
first knowing which draw is the aircraft, which is the problem it was supposed
to avoid.

## The approach: classify in view space

The patched vertex shader can recover view-space position from `gl_Position`
for three divides, because for X-Plane's perspective projection

    view.z = -clip.w
    view.x =  clip.x / proj[0]
    view.y =  clip.y / proj[5]

and `proj[0]` / `proj[5]` are already carried in the push block that the motion
vector work uses.

View space is a **frame shared by every draw in the frame**. The plugin knows
where the aircraft is relative to the camera, so it publishes the aircraft's
view-space transform each frame, and the shader asks a purely spatial question:

    vec3 local = aircraftInv * viewPos;      // into airframe coordinates
    ivec3 cell = ivec3(floor((local - gridMin) / cellSize));
    int   part = cell.x + cell.y*NX + cell.z*NX*NY;
    if (part is occupied) viewPos = partXform[part] * viewPos;

This resolves three problems at once:

- **No object identity.** A vertex belongs to the airframe if it is spatially
  inside the airframe. Which draw produced it is irrelevant.
- **Animation composes for free.** The displacement is applied *after*
  X-Plane's model, animation and skinning transforms have run, so gear and
  control surfaces arrive correctly placed and stay attached to their fragment.
- **Shadow and reflection passes follow automatically**, because they run
  through the same patched shaders with the same maths.

## Fragment size, not fragment count

The eye judges fragment SIZE. A piece reads as debris at roughly human scale —
a door panel, a control surface, a section of skin — which is about 1.2 m.

Count follows from size and aircraft dimensions:

| aircraft   | bounding box      | 1.2 m cells | occupied (~5–10%) |
|------------|-------------------|-------------|-------------------|
| Cessna 172 | 8 × 11 × 3 m      | ~1,500      | ~120–200          |
| A320       | 38 × 36 × 12 m    | ~11,000     | ~600–1,100        |
| 747        | 70 × 65 × 19 m    | ~60,000     | ~1,200–2,500      |

Aircraft bounding boxes are mostly air — wings are thin sheets in a large
volume — so genuine occupancy is a few percent. Classification runs over the
full grid arithmetically (empty cells cost nothing, no vertex lands in them),
while transforms are allocated only for occupied cells.

Cell size scales with the aircraft's bounding box so fragment size stays
constant across aircraft. No per-model authoring.

**Target: 800–2000 fragments for an airliner.** Below ~300 the result reads as
an object splitting into slabs. Above ~3000 the returns diminish and the
wreckage begins to look like sand rather than structure.

More fragments also removes what would otherwise be the worst artefact:
X-Plane models are hollow shells, so a detached wing at low fragment counts is
a visibly empty box. At 1.2 m, a fragment is small enough that having no
interior reads as debris rather than as broken rendering.

## Constraints are what make it look like destruction

Several hundred independent rigid bodies is confetti. Destruction reads as
destruction because parts stay attached until they don't, and load transfers to
whatever is still joined.

Each occupied cell is joined to its six-connected neighbours by a distance
constraint with a break threshold. Position-based dynamics, a few iterations
per step. When a constraint breaks its load redistributes, so failure
propagates: the nose crushes, stress runs aft, the tail shears off a moment
later.

Cost is not a concern. 2000 parts is about 6000 constraints; at 10 iterations
that is 60k constraint solves per step, microseconds on one core.

The tuning risk is real though: past a few thousand fragments with weak joints
an airframe stops shearing and starts pouring. If it looks wrong, the fix is
stiffness and break thresholds — not more fragments.

## Discovering which cells contain geometry

Constraints need occupancy, and neither the plugin nor the layer can see the
mesh. The GPU can answer: once the storage buffer exists, the patched vertex
shader sets a bit for its cell, and one frame of rendering yields the airframe's
occupancy grid, discovered by the geometry itself. Works for any aircraft with
no authoring.

## Ground

`XPLMProbeTerrainXYZ` at the crash site gives a height and normal. One plane is
enough for a debris field. Each fragment's lowest point is clamped against it,
normal velocity killed, friction applied, and contact offset producing torque.

## Scope

IN: visual fragmentation, breakable constraints, ground settling, crash
trigger, freezing the airframe while wreckage settles.

OUT: aerodynamic feedback from damage, fragment-to-fragment collision,
fragments colliding with scenery, persistence across flights, multiplayer.

## Risks, highest first

1. **View-space reconstruction may not hold** for every pass. Shadow passes may
   use orthographic projections where `view.z = -clip.w` is false, and the TAA
   jitter offset perturbs `clip.xy`. This is the load-bearing assumption and is
   probed first.
2. **Adding a descriptor set to every graphics pipeline** is more invasive than
   anything the layer does today — the existing work only appends to pipeline
   layouts and patches shader code. The set must also be bound before every
   draw without disturbing X-Plane's own descriptor usage.
3. **`override_planepath` is sharp-edged.** It locked the camera during the
   motion-vector work and must be cleared on every exit path.
4. **Constraint tuning** decides whether the result looks like structural
   failure or like sand.
