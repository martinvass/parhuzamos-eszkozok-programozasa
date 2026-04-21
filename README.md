# Párhuzamos eszközök programozása

Miskolci Egyetem — Párhuzamos eszközök programozása tárgy féléves anyagai.

## Beadandó — Gauss-elimináció OpenCL-lel

A program egy lineáris egyenletrendszert (**A · x = b**) old meg Gauss-eliminációval. Az előre-elimináció lépéseit a GPU végzi párhuzamosan: minden pivot lépésnél a pivot alatti sorok frissítését külön-külön work itemek dolgozzák fel egyszerre.

### Fordítás és futtatás

```bash
cd beadandó
make
./gen 1000 42 > matrix.txt        # 1000×1000 tesztmátrix generálása
./gauss_ocl matrix.txt            # megoldás
./gauss_ocl matrix.txt -b         # benchmark mód (csak futásidő)
```

## Technológia

- Nyelv: **C**
- Párhuzamosítás: **OpenCL**
- Platformok: macOS (Apple M1) / Linux / Windows

## Mérések

A beadandóhoz készített mérések, diagramok és a jegyzőkönyv a beadandó/meresek/ mappában találhatóak.
