// Sprite stacking: the voxel model drawn as a pile of flat pictures, one per 0.5 cm of height, each the
// model's slice at that height seen from above.  Drawn bottom to top on a 2D canvas, each a little higher on
// the screen and squashed by the view's tilt, the pile looks solid.  No 3D engine is needed to draw it.
// Exposes window.SpriteStack.  Needs YamabikoRig's Voxels (rig.js) for the input.
(function () {
  const RES = .5;                                  // cm per pixel and per layer

  // cells: [[x, y, z, color, jitter], ...] in units of `unit` cm.  Returns {layers: [canvas], x0, z0, y0}
  function build(sets) {
    // sets: [{voxels, unit}]  (voxels: a Voxels; unit: its size in cm)
    let xs = [Infinity, -Infinity], ys = [Infinity, -Infinity], zs = [Infinity, -Infinity];
    const cells = [];
    for (const { voxels, unit } of sets) {
      const k = Math.round(unit / RES);
      for (const [x, y, z, color, jitter] of voxels.cells.values()) {
        cells.push([x * k, y * k, z * k, k, color, jitter, voxels, x, y, z]);
        xs = [Math.min(xs[0], x * k), Math.max(xs[1], x * k + k)];
        ys = [Math.min(ys[0], y * k), Math.max(ys[1], y * k + k)];
        zs = [Math.min(zs[0], z * k), Math.max(zs[1], z * k + k)];
      }
    }
    const W = xs[1] - xs[0], D = zs[1] - zs[0], H = ys[1] - ys[0];
    const layers = Array.from({ length: H }, () => { const c = document.createElement("canvas"); c.width = W; c.height = D; return c; });
    const ctx = layers.map(c => c.getContext("2d"));
    const col = new THREE.Color();
    let seed = cells.length * 7919;
    const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (const [px, py, pz, k, color, jitter, voxels, x, y, z] of cells) {
      // a block with nothing on it shows its top (lit); one under another shows its side (darker)
      const top = !voxels.cells.has(`${x},${y + 1},${z}`);
      col.setHex(color).multiplyScalar((1 - jitter + 2 * jitter * rand()) * (top ? 1 : .74));
      for (let j = 0; j < k; j++) {
        const g = ctx[py + j - ys[0]];
        g.fillStyle = "#" + col.getHexString();
        g.fillRect(px - xs[0], pz - zs[0], k, k);
      }
    }
    return { layers, x0: xs[0] * RES, z0: zs[0] * RES, y0: ys[0] * RES, w: W * RES, d: D * RES, h: H * RES };
  }

  // a view: the screen point of the world point (cm), and the drawing of stacks
  class View {
    constructor(canvas) {
      this.c = canvas; this.g = canvas.getContext("2d");
      this.yaw = 0; this.pitch = THREE.MathUtils.degToRad(48); this.scale = 5; this.cx = 74; this.cz = -14; this.cy = 4;
      this.drawn = 0;
    }
    project(x, y, z) {
      const dx = x - this.cx, dz = z - this.cz, c = Math.cos(this.yaw), s = Math.sin(this.yaw);
      const u = dx * c - dz * s, w = dx * s + dz * c;
      return [this.c.width / 2 + u * this.scale, this.c.height / 2 + (w * Math.sin(this.pitch) - (y - this.cy) * Math.cos(this.pitch)) * this.scale];
    }
    clear(color) {
      const g = this.g; g.setTransform(1, 0, 0, 1, 0, 0); g.imageSmoothingEnabled = false;   // crisp blocks
      g.fillStyle = color; g.fillRect(0, 0, this.c.width, this.c.height); this.drawn = 0;
    }
    // draw layer `i` of `stack`, its origin moved to (ox, oz) and turned by `angle` about (ax, az) (world cm)
    layer(stack, i, ox = 0, oz = 0, angle = 0, ax = 0, az = 0) {
      const g = this.g, y = stack.y0 + i * RES;
      const sp = Math.sin(this.pitch), cp = Math.cos(this.pitch);
      const step = RES * cp * this.scale, reps = Math.max(1, Math.ceil(step));
      for (let r = 0; r < reps; r++) {
        const lift = (y - this.cy) * cp * this.scale + step * r / reps;
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.translate(this.c.width / 2, this.c.height / 2 - lift);
        g.scale(this.scale, this.scale * sp);
        g.rotate(this.yaw);
        g.translate(-this.cx + ox, -this.cz + oz);
        if (angle) { g.translate(ax, az); g.rotate(angle); g.translate(-ax, -az); }
        g.drawImage(stack.layers[i], stack.x0, stack.z0, stack.w, stack.d);
        this.drawn++;
      }
    }
  }

  window.SpriteStack = { build, View, RES };
})();
