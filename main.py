import sys
import os
import traceback


def _show_error(tb):
    # persist the error for later reading (best effort)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("/data/local/tmp/mountain_rider_error.log", os.path.join(here, "error.log")):
        try:
            with open(p, "w") as f:
                f.write(tb)
        except Exception:
            pass
    # try to render the traceback on screen so it can be screenshotted
    try:
        import pygame
    except Exception:
        sys.stderr.write(tb)
        return
    try:
        pygame.init()
        screen = pygame.display.set_mode((960, 540))
        font = pygame.font.SysFont("monospace", 20)
        y = 10
        for ln in tb.splitlines()[-26:]:
            screen.blit(font.render(ln, True, (235, 235, 235)), (10, y))
            y += 21
        pygame.display.flip()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    sys.exit()
            pygame.time.wait(40)
    except Exception:
        sys.stderr.write(tb)


def main():
    try:
        import mountain_rider
        mountain_rider.Game().run()
    except Exception:
        _show_error(traceback.format_exc())


if __name__ == "__main__":
    main()