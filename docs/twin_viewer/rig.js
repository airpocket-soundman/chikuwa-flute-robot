// The voxel model of やまびこ1号, shared by the one-rig page (index.html) and the 32-rig training viewer
// (fleet.html).  1 voxel = 1 cm.  x: along the rail (left = mouthpiece), y: up, z: toward the viewer.
// Needs THREE (r128) loaded first.  Exposes window.YamabikoRig.
(function () {
  const C = {
    alu: 0xb9bdc3, aluDark: 0x8e939a, white: 0xf1efe9, black: 0x26272b, bolt: 0x1c1c1f,
    blue: 0x2c33b8, servo: 0x2f6fe0, rod: 0xd7d9dc, steel: 0xc9ccd1, actGray: 0x9da1a8,
    pcbBlue: 0x1f5fbf, pcbGreen: 0x1e8a45, chip: 0x202226, gold: 0xd8b25a, cap: 0x2c2c30,
  };

  class Voxels {
    constructor() { this.cells = new Map(); }
    set(x, y, z, color, jitter = .06) { this.cells.set(`${x},${y},${z}`, [x, y, z, color, jitter]); }
    box(x0, x1, y0, y1, z0, z1, color, jitter) {
      for (let x = x0; x < x1; x++) for (let y = y0; y < y1; y++) for (let z = z0; z < z1; z++) this.set(x, y, z, color, jitter);
      return this;
    }
    // a cylinder along x: centre (cy, cz), radius r, hollow inside ri
    cylX(x0, x1, cy, cz, r, color, ri = -1, jitter) {
      for (let x = x0; x < x1; x++) for (let y = Math.floor(cy - r); y <= Math.ceil(cy + r); y++)
        for (let z = Math.floor(cz - r); z <= Math.ceil(cz + r); z++) {
          const d = Math.hypot(y + .5 - cy, z + .5 - cz);
          if (d <= r && d > ri) this.set(x, y, z, color, jitter);
        }
      return this;
    }
    // a cylinder along y: centre (cx, cz)
    cylY(y0, y1, cx, cz, r, color, ri = -1, jitter) {
      for (let y = y0; y < y1; y++) for (let x = Math.floor(cx - r); x <= Math.ceil(cx + r); x++)
        for (let z = Math.floor(cz - r); z <= Math.ceil(cz + r); z++) {
          const d = Math.hypot(x + .5 - cx, z + .5 - cz);
          if (d <= r && d > ri) this.set(x, y, z, color, jitter);
        }
      return this;
    }
    // every voxel its own cube (fine for one rig; shadows and the look of separate blocks)
    mesh(origin = [0, 0, 0]) {
      const list = [...this.cells.values()];
      const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(.96, .96, .96),
        new THREE.MeshStandardMaterial({ roughness: .78, metalness: .05 }), list.length);
      const m = new THREE.Matrix4(), col = new THREE.Color();
      let seed = list.length * 9301;
      const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      list.forEach(([x, y, z, color, jitter], i) => {
        m.makeTranslation(x + .5 - origin[0], y + .5 - origin[1], z + .5 - origin[2]);
        mesh.setMatrixAt(i, m);
        col.setHex(color).multiplyScalar(1 - jitter + 2 * jitter * rand());
        mesh.setColorAt(i, col);
      });
      mesh.castShadow = mesh.receiveShadow = true;
      return mesh;
    }
    // a coarser copy: every k x k x k block one voxel (its first voxel's colour), drawn scaled by k
    downsample(k) {
      const out = new Voxels();
      for (const [x, y, z, color, jitter] of this.cells.values()) {
        const key = `${Math.floor(x / k)},${Math.floor(y / k)},${Math.floor(z / k)}`;
        if (!out.cells.has(key)) out.cells.set(key, [Math.floor(x / k), Math.floor(y / k), Math.floor(z / k), color, jitter]);
      }
      return out;
    }
    // one geometry of only the faces nobody covers (for drawing many rigs at once), vertex colours
    geometry(origin = [0, 0, 0]) {
      const pos = [], nor = [], col = [];
      const FACES = [
        [[1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]], [[-1, 0, 0], [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]],
        [[0, 1, 0], [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]]], [[0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
        [[0, 0, 1], [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]]], [[0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]],
      ];
      const c = new THREE.Color();
      let seed = this.cells.size * 9301;
      const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      for (const [x, y, z, color, jitter] of this.cells.values()) {
        c.setHex(color).multiplyScalar(1 - jitter + 2 * jitter * rand());
        for (const [n, quad] of FACES) {
          if (this.cells.has(`${x + n[0]},${y + n[1]},${z + n[2]}`)) continue;
          const shade = n[1] === 1 ? 1 : n[1] === -1 ? .6 : (n[0] ? .82 : .9);   // a little baked light
          for (const k of [0, 1, 2, 0, 2, 3]) {
            const v = quad[k];
            pos.push(x + v[0] - origin[0], y + v[1] - origin[1], z + v[2] - origin[2]);
            nor.push(...n); col.push(c.r * shade, c.g * shade, c.b * shade);
          }
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
      g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
      return g;
    }
  }

  // the slide: STROKE cm of travel; the carriage moves between the whistle's end (x 67) and the actuator's
  // holder (x 118).  The twin's position 0 (homed, the plunger out, lowest note) is the carriage at the right.
  const STROKE = 22, SLIDE_HOME = 24;
  const FAN = [4, 0, -38];
  const LEDS = { y: 15.5, z: -12.2, shut: 2.5, open: 5.5 };

  function build() {
    const fixed = new Voxels();
    // aluminium extrusion along the whole rig, with two slots on the front and top
    fixed.box(-2, 156, -2, 3, -4, 4, C.alu, .04);
    fixed.box(-2, 156, 0, 1, 3, 4, C.aluDark, .02);
    fixed.box(-2, 156, 2, 3, -1, 1, C.aluDark, .02);
    // white clamp blocks on the rail, black bolt heads on the front
    const clamp = (x0, x1, y1, zb = -5, zf = 5) => {
      fixed.box(x0, x1, -2, y1, zb, zf, C.white, .03);
      for (const bx of [x0 + 1, x1 - 2]) fixed.box(bx, bx + 1, 0, 1, zf, zf + 1, C.bolt, .02);
    };
    clamp(30, 42, 12);                  // the tube holder
    clamp(66, 71, 4);                   // board stays
    clamp(91, 96, 4);
    clamp(118, 130, 12);                // the actuator's holder
    // slide whistle: white mouthpiece (its window on top), blue tube, white collar at its end
    fixed.cylX(10, 22, 7.5, 0, 2.6, C.white, -1, .03);
    fixed.box(9, 12, 5, 10, -3, 3, C.white, .03);
    fixed.cylX(22, 64, 7.5, 0, 2.6, C.blue, -1, .05);
    fixed.box(14, 18, 9, 10, -1, 1, C.black, .02);
    fixed.cylX(64, 67, 7.5, 0, 2.9, C.white, -1, .03);
    fixed.box(31, 41, 10, 12, -4, 4, C.white, .03);              // tube holder's saddle
    // the air path: fan -> valve box (behind the whistle) -> an L-shaped white pipe -> mouthpiece
    fixed.box(-1, 9, 6, 15, -20, -11, C.white, .03);
    fixed.box(2, 6, 6, 10, -11, 2, C.white, .03);
    fixed.box(2, 10, 6, 10, -2, 2, C.white, .03);
    // centrifugal blower behind the valve box
    fixed.cylY(0, 5, FAN[0], FAN[2], 15, C.white, -1, .03);
    fixed.box(0, 8, 6, 14, -26, -20, C.white, .03);
    fixed.cylY(5, 17, FAN[0], FAN[2], 13, C.black, 9.5, .05);
    fixed.cylY(16, 17, FAN[0], FAN[2], 9.6, C.black, 7, .05);
    // servo (SG90) on the valve box's top (its horn and the valve are inside the box)
    fixed.box(1, 7, 15, 21, -18, -14, C.servo, .04);
    // microphone board on its white bracket, beside the tube holder
    fixed.box(28, 30, 3, 24, -6, -4, C.white, .03);
    fixed.box(27, 31, 20, 25, -4, -3, C.pcbGreen, .05);
    fixed.box(28, 30, 22, 24, -3, -2, C.chip, .02);
    // Arduino UNO Q (blue) with the board stack (green) on stand-offs, behind the whistle
    fixed.box(70, 98, 3, 4, -23, -11, C.pcbBlue, .05);
    for (const [x, z] of [[71, -22], [96, -22], [71, -12], [96, -12]]) fixed.box(x, x + 1, 4, 7, z, z + 1, C.gold, .02);
    fixed.box(68, 100, 7, 8, -24, -10, C.pcbGreen, .05);
    fixed.box(72, 80, 8, 9, -16, -12, C.chip, .02);
    fixed.box(84, 87, 8, 13, -22, -19, C.cap, .03);
    fixed.box(88, 90, 8, 12, -22, -20, C.cap, .03);
    fixed.box(92, 98, 8, 11, -23, -18, C.chip, .03);
    fixed.box(80, 84, 8, 10, -23, -21, C.white, .02);
    fixed.box(70, 74, 8, 10, -23, -21, C.white, .02);
    // linear actuator: grey body, black end cap at the far right
    fixed.box(128, 160, 6, 11, -3, 3, C.actGray, .04);
    fixed.box(160, 164, 5, 12, -4, 4, C.black, .04);
    fixed.box(134, 136, 3, 6, -2, 2, C.white, .03);

    // fan rotor: dark blades inside the housing, about its own axis (origin at the axis)
    const rotor = new Voxels();
    for (let a = 0; a < 24; a++) {
      const th = a / 24 * 2 * Math.PI;
      for (let r = 5; r <= 9; r++) {
        const x = Math.round(Math.cos(th + r * .05) * r), z = Math.round(Math.sin(th + r * .05) * r);
        rotor.box(x, x + 1, 6, 15, z, z + 1, 0x4a4d55, .08);
      }
    }
    rotor.cylY(14, 15, 0, 0, 3.2, 0x8a8f99, -1, .05);

    // the slide: rod in the tube, the carriage, the actuator rod (placed at x = SLIDE_HOME + offset)
    const slide = new Voxels();
    slide.box(0, 56, 7, 8, 0, 1, C.rod, .03);
    slide.box(56, 64, 4, 13, 0, 5, C.white, .03);
    slide.box(57, 58, 5, 6, 5, 6, C.bolt, .02).box(62, 63, 5, 6, 5, 6, C.bolt, .02);
    slide.box(64, 114, 8, 9, 0, 1, C.steel, .02);
    return { fixed, rotor, slide };
  }

  // the carriage's x offset for the twin's plunger position (0 = homed at the right, 1 = pushed in)
  const slideX = (position) => SLIDE_HOME + (1 - Math.min(Math.max(position, 0), 1)) * STROKE;

  window.YamabikoRig = { C, Voxels, build, STROKE, SLIDE_HOME, FAN, LEDS, slideX };
})();
