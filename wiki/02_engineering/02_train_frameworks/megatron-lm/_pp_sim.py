# PP schedule simulator: computes exact space-time slots from the Megatron algorithm.
def get_schedule_table(m, v, N):
    table = []
    for mn in range(0, m, N):
        if mn + N >= m:
            for c in range(v):
                for mb in range(mn, m):
                    table.append((mb, c))
        else:
            for c in range(v):
                for mb in range(mn, mn + N):
                    table.append((mb, c))
    return table


def build_1f1b(p, m):
    devs = {}
    for r in range(p):
        w = min(p - 1 - r, m)
        seq = []
        for j in range(w):
            seq.append(('F', j, 0))
        for i in range(m - w):
            seq.append(('F', i + w, 0))
            seq.append(('B', i, 0))
        for j in range(m - w, m):
            seq.append(('B', j, 0))
        devs[r] = seq
    return devs


def build_vpp(p, m, v, N, extra_warmup=0):
    table = get_schedule_table(m, v, N)
    total = m * v
    mb_t = [x[0] for x in table]
    ch_t = [x[1] for x in table]
    devs = {}
    warmups = {}
    for r in range(p):
        w = 2 * (p - 1 - r) + (v - 1) * N + extra_warmup
        if w >= total:
            w = total
        warmups[r] = w
        seq = []
        for k in range(w):
            seq.append(('F', mb_t[k], ch_t[k]))
        for k in range(total - w):
            fk = k + w
            seq.append(('F', mb_t[fk], ch_t[fk]))
            seq.append(('B', mb_t[k], v - 1 - ch_t[k]))
        for k in range(total - w, total):
            seq.append(('B', mb_t[k], v - 1 - ch_t[k]))
        devs[r] = seq
    return devs, warmups, table


def simulate(devs, p, v):
    slot = {}
    seqs = {}
    for r, seq in devs.items():
        seqs[r] = [(k, mb, c, r) for (k, mb, c) in seq]
        for op in seqs[r]:
            slot[op] = 1

    def deps(op):
        kind, mb, c, r = op
        d = []
        if kind == 'F':
            if r > 0:
                d.append(('F', mb, c, r - 1))
            elif r == 0 and c > 0:
                d.append(('F', mb, c - 1, p - 1))
        else:
            if r < p - 1:
                d.append(('B', mb, c, r + 1))
            elif r == p - 1 and c < v - 1:
                d.append(('B', mb, c + 1, 0))
            elif r == p - 1 and c == v - 1:
                d.append(('F', mb, v - 1, p - 1))
            d.append(('F', mb, c, r))
        return [x for x in d if x in slot]

    for _ in range(100000):
        changed = False
        for r in sorted(seqs):
            prev = 0
            for op in seqs[r]:
                s = prev + 1
                for dep in deps(op):
                    s = max(s, slot[dep] + 1)
                if s != slot[op]:
                    slot[op] = s
                    changed = True
                prev = slot[op]
        if not changed:
            break
    return slot, seqs


def render(slot, seqs, p, v, label):
    # compact notation for v==2: f=fwd chunk0, F=fwd chunk1, b=bwd chunk0, B=bwd chunk1
    makespan = max(slot.values())
    print("===", label, "  makespan =", makespan, "===")
    cw = 3 if v > 1 else 4
    hdr = "slot".ljust(cw)
    for t in range(1, makespan + 1):
        hdr += str(t).ljust(cw)
    print(hdr)
    for r in range(p):
        cells = ['..'] * makespan
        for op in seqs[r]:
            kind, mb, c, rr = op
            t = slot[op]
            if v > 1:
                letter = {('F', 0): 'f', ('F', 1): 'F', ('B', 0): 'b', ('B', 1): 'B'}[(kind, c)]
                cells[t - 1] = letter + str(mb)
            else:
                cells[t - 1] = kind + str(mb)
        line = ("Dev" + str(r)).ljust(cw)
        for cell in cells:
            line += cell.ljust(cw)
        print(line)
    for r in range(p):
        ops = len(seqs[r])
        idle = makespan - ops
        idle_slots = [t for t in range(1, makespan + 1)
                      if all(slot[op] != t for op in seqs[r])]
        print("  Dev%d: %d ops, %d idle, idle slots = %s" % (r, ops, idle, idle_slots))
    print()


# --- verification against hand-computed cases ---
d = build_1f1b(4, 8)
s, q = simulate(d, 4, 1)
render(s, q, 4, 1, "1F1B  p=4 m=8 (verify: expect makespan 22)")

d, w, t = build_vpp(2, 4, 2, 2)
s, q = simulate(d, 2, 2)
render(s, q, 2, 2, "VPP  p=2 v=2 m=4 (verify: expect makespan 18)")

# --- target diagrams: 4 stages ---
d, w, t = build_vpp(4, 8, 2, 4)
print("VPP p=4 schedule table:", t)
print("VPP p=4 warmups:", w)
s, q = simulate(d, 4, 2)
render(s, q, 4, 2, "VPP  p=4 v=2 m=8 N=4  (plain interleaved)")

d, w, t = build_vpp(4, 8, 2, 4, extra_warmup=1)
print("combined-1F1B-on-VPP p=4 warmups (+1):", w)
s, q = simulate(d, 4, 2)
render(s, q, 4, 2, "combined-1F1B on VPP  p=4 v=2 m=8 N=4  (warmup +1)")
