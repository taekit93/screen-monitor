"""녹화 영상으로 감지 설정을 검증하는 도구.

GUI(main.py)와 **똑같은 판정 클래스**(watch.RuneWatch / CharWatch)를 돌린다.
그래서 여기서 나온 사건 목록이 곧 "실제로 그때 소리가 울렸을 횟수" 다.

실행 예)
  .venv\\Scripts\\python.exe tools\\analyze_video.py "샘플 영상\\a.mp4" ^
      --roi 22,42,178,60 --what rune --dump out\\rune

옵션
  --roi x,y,w,h     영상 안에서 미니맵 영역 (생략하면 전체 프레임)
  --what rune|char  무엇을 검사할지 (기본 rune)
  --allow x,y,w,h   캐릭터 허용 영역 (ROI 기준 상대 좌표). --what char 에 필요
  --fps N           초당 N장 검사 (기본 4 = GUI 기본 감지 주기 0.25초와 같음)
  --settings P      settings.json 을 읽어 그 설정 그대로 검증
  --dump DIR        알림이 울린 장면을 모아 montage.png 로 저장
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfgmod                    # noqa: E402
from watch import CharWatch, RuneWatch     # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--roi", default="")
    ap.add_argument("--what", default="rune", choices=("rune", "char"))
    ap.add_argument("--allow", default="")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--settings", default="")
    ap.add_argument("--dump", default="")
    return ap.parse_args()


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main() -> int:
    a = parse_args()
    cfg = cfgmod.load(a.settings) if a.settings else cfgmod.AppConfig()
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    allow = tuple(int(v) for v in a.allow.split(",")) if a.allow else None

    if a.what == "rune":
        rule = cfg.rune()
        watch = RuneWatch(cfg.rune_hold, cfg.rune_repeat, cfg.rune_repeat_interval)
        print(f"룬 검사: {cfg.rune_hold}초 이상 계속 보이면 알림, "
              f"{cfg.rune_repeat_interval:.0f}초마다 재알림"
              f"{'' if cfg.rune_repeat else ' (재알림 꺼짐)'}")
    else:
        if not allow:
            print("--allow x,y,w,h 를 지정해야 캐릭터 이탈을 판정할 수 있습니다.")
            return 1
        rule = cfg.char()
        watch = CharWatch(allow, cfg.char_hold, cfg.char_out_after,
                          cfg.char_lost_after, cfg.char_repeat_interval)
        print(f"캐릭터 검사: 허용 영역 {allow}, 밖에 {cfg.char_out_after}초 / "
              f"안 보인 지 {cfg.char_lost_after}초면 알림")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("영상을 열 수 없습니다:", a.video)
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, int(round(src_fps / max(0.01, a.fps))))

    alerts, shots, notes = [], [], []
    n_samples = raw_hits = 0
    idx = 0

    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                if roi:
                    x, y, w, h = roi
                    frame = frame[y:y + h, x:x + w]
                t = idx / src_fps
                n_samples += 1
                dets = rule.detect(frame)
                raw_hits += bool(dets)
                ev = watch.update(dets, t)
                if ev is not None:
                    if ev.is_alert:
                        alerts.append((t, ev))
                        if ev.first:
                            shots.append((t, frame.copy(), ev))
                    else:
                        notes.append((t, ev))
        idx += 1
    cap.release()

    firsts = [(t, e) for t, e in alerts if e.first]
    print(f"\n검사 샘플 {n_samples}개 (원본 {src_fps:.2f}fps, 초당 {a.fps}장)")
    print(f"색·모양만 맞은 샘플 {raw_hits}개 "
          f"({raw_hits / max(1, n_samples) * 100:.1f}%)  "
          f"— 시간 조건까지 통과해야 알림입니다")
    print(f"알림 소리 {len(alerts)}회 (새로 생긴 일 {len(firsts)}건, "
          f"이어지는 중 재알림 {len(alerts) - len(firsts)}회)")
    for t, e in firsts:
        extra = f" {e.pos}" if e.pos else ""
        print(f"   {mmss(t)}  {e.kind}{extra}")

    if a.dump and shots:
        os.makedirs(a.dump, exist_ok=True)
        tiles = []
        for t, f, e in shots[:40]:
            v = f.copy()
            if allow:
                x, y, w, h = allow
                cv2.rectangle(v, (x, y), (x + w - 1, y + h - 1), (255, 160, 60), 1)
            if e.pos:
                cv2.drawMarker(v, e.pos, (0, 255, 0), cv2.MARKER_CROSS, 9, 1)
            sc = max(1, int(360 / max(v.shape[1], 1)))
            v = cv2.resize(v, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            v = cv2.copyMakeBorder(v, 16, 2, 2, 2, cv2.BORDER_CONSTANT)
            cv2.putText(v, f"{mmss(t)} {e.kind}", (4, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(v)
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1],
                                    cv2.BORDER_CONSTANT) for t in tiles]
        out = os.path.join(a.dump, "montage.png")
        cv2.imencode(".png", np.vstack(tiles))[1].tofile(out)
        print("\n몽타주 저장:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
