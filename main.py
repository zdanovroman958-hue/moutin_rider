import sys
import os
import traceback

import mountain_rider


def main():
    try:
        mountain_rider.Game().run()
    except Exception as exc:
        tb = traceback.format_exc()
        # persist the error so it can be read later
        for path in ("/data/local/tmp/mountain_rider_error.log", "mountain_rider_error.log"):
            try:
                with open(path, "w") as f:
                    f.write(tb)
            except Exception:
                pass
        # try to show the traceback on screen so it can be screenshotted
        try:
            pygame = mountain_rider.pygame
            pygame.init()
            screen = pygame.display.set_mode((mountain_rider.W, mountain_rider.H))
            font = pygame.font.SysFont("monospace", 20)
            lines = tb.splitlines()[-26:]
            y = 8
            for ln in lines:
                surf = font.render(ln, True, (235, 235, 235))
                screen.blit(surf, (8, y))
                y += 20
            pygame.display.flip()
            while True:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        sys.exit()
                pygame.time.wait(40)
        except Exception:
            sys.stderr.write(tb)
            raise


if __name__ == "__main__":
    main()