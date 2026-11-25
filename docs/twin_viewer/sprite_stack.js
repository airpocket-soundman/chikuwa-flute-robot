// Sprite stacking: the voxel model drawn as a pile of flat pictures, one per 0.5 cm of height, each the
// model's slice at that height seen from above.  The pictures are thin sheets stacked in the 3D view and
// drawn with the same perspective camera as the voxel model, so the pile looks solid and keeps the
// perspective (near things bigger, far things smaller).
// Exposes window.SpriteStack.  Needs three.js and YamabikoRig's Voxels (rig.js) for the input.
(function () {
  const RES = .5;                                  // cm per pixel and per layer
  const SHEETS = 3;                                // sheets per layer, so no gaps show between layers

  // sets: [{voxels, unit}] (unit: the voxels' size in cm) -> {layers: [canvas], x0, y0, z0, w, d}
  function build(sets) {
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

  // the stack as a group of flat sheets (in the voxels' own coordinates), drawn as the pictures are: no light
  function group(stack) {
    const g = new THREE.Group();
    stack.layers.forEach((canvas, i) => {
      const tex = new THREE.CanvasTexture(canvas);
      tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestMipmapNearestFilter;
      const mat = new THREE.MeshBasicMaterial({ map: tex, alphaTest: .5, side: THREE.DoubleSide });
      const geo = new THREE.PlaneGeometry(stack.w, stack.d);
      for (let s = 0; s < SHEETS; s++) {
        const m = new THREE.Mesh(geo, mat);
        m.rotation.x = -Math.PI / 2;
        m.position.set(stack.x0 + stack.w / 2, stack.y0 + (i + (s + .5) / SHEETS) * RES, stack.z0 + stack.d / 2);
        g.add(m);
      }
    });
    g.userData.sheets = stack.layers.length * SHEETS;
    return g;
  }

  // the whole rig as stacks in a scene of its own: the fixed parts, the slide and the fan's rotor as
  // separate stacks so they move; update(slideX, fanAngle, valveOpen) every frame.
  function rigScene({ fixedSets, slide, rotor, FAN, LEDS, background = 0x17191d }) {
    const t0 = performance.now();
    const scene = new THREE.Scene(); scene.background = new THREE.Color(background);
    const stacks = { fixed: build(fixedSets), slide: build([{ voxels: slide, unit: 1 }]), rotor: build([{ voxels: rotor, unit: 1 }]) };
    const fixed = group(stacks.fixed), slider = group(stacks.slide), rotorG = new THREE.Group();
    const r = group(stacks.rotor); r.position.set(-.5, 0, -.5); rotorG.add(r);    // the rotor's axis at its voxels' centre
    rotorG.position.set(FAN[0], 0, FAN[2]);
    scene.add(fixed, slider, rotorG);
    const led = (x, lit, dim) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1, 1.6), new THREE.MeshBasicMaterial({ color: dim }));
      m.position.set(x, LEDS.y, LEDS.z); scene.add(m);
      return { m, lit: new THREE.Color(lit), dim: new THREE.Color(dim) };
    };
    const shut = led(LEDS.shut, 0xff2a1a, 0x4a1210), open = led(LEDS.open, 0x22ff55, 0x103f1a);
    const sheets = [fixed, slider, r].reduce((a, g) => a + g.userData.sheets, 0);
    const pixels = Object.values(stacks).reduce((a, q) => a + q.layers.length * q.layers[0].width * q.layers[0].height, 0);
    return {
      scene, sheets, pixels, layers: Object.values(stacks).reduce((a, q) => a + q.layers.length, 0), ms: performance.now() - t0,
      update(slideX, fanAngle, valveOpen) {
        slider.position.x = slideX; rotorG.rotation.y = fanAngle;
        shut.m.material.color.copy(valveOpen ? shut.dim : shut.lit);
        open.m.material.color.copy(valveOpen ? open.lit : open.dim);
      },
    };
  }

  window.SpriteStack = { build, group, rigScene, RES, SHEETS };
})();
