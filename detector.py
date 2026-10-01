"""미니맵 아이콘 감지 규칙.

규칙은 하나뿐이다. 아이콘을 다음 네 가지로 찾는다.

  1. 색       HSV 범위
  2. 크기/모양  픽셀 수, 변 길이, 가로세로 비율, 채움비
  3. 밝은 테두리  덩어리 바깥 2px 띠에서 밝은 픽셀의 비율
  4. 흰 테두리   덩어리에 딱 붙은 1px 띠에서 **진짜 흰색** 픽셀의 비율

4번이 오탐을 거의 다 잡아낸다. 메이플 미니맵 아이콘(룬, 캐릭터, 다른 유저)에는
흰 테두리가 둘러져 있지만, 아이콘과 같은 색으로 그려진 지형 그림에는 없다.
지형 얼룩도 '밝기'는 밝을 수 있어서 3번만으로는 부족하고(샘플 영상에서 가짜의
절반이 통과), 채도까지 낮은 **흰색**을 요구하면 걸러진다.

  샘플 영상 2편에서 잰 값 (덩어리 단위, 진짜 572개 / 가짜 41개)
    3번 밝은 테두리 0.28 이상 → 진짜 74.5% 통과, 가짜 48.8% 통과
    4번 흰 테두리  0.12 이상 → 진짜 79.0% 통과, 가짜  0.0% 통과

여기에 **마름모 테두리 훑기**(ring_scan)를 하나 더 얹었다. 위 규칙이 못 잡는
두 경우를 위해서다.

  - 룬이 보라 지형에 붙어 있을 때. 둘이 한 덩어리로 잡혀 크기/비율에서 떨어진다.
  - 미니맵이 작고 영상이 압축될 때. 색 정보가 2x2 로 뭉개져서 흰 테두리가
    회색(채도 40~90, 밝기 130~170)으로 번진다. '진짜 흰색' 이 거의 안 남는다.

덩어리를 보지 않고, 보라 픽셀 하나하나를 중심으로 삼아 '이 점이 룬의 한가운데라면
마름모 테두리 자리가 밝은가' 를 반지름별로 잰다. 테두리는 네 변으로 나눠
**가장 어두운 변**을 점수로 쓴다. 지형 얼룩은 한두 변만 밝아서 여기서 떨어지고,
룬은 지형 위에 그려지므로 붙어 있어도 네 변이 다 산다.

  카르시온 영상 8편(1시간 37분, 룬 10개, 그중 4개가 보라 지형 위)
    기존 규칙만      룬 0개, 사냥터 오탐 0
    + 마름모 훑기    룬 10개, 사냥터 오탐 0
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    fill: float = 0.0        # 바운딩 박스 대비 채워진 비율
    ring: float = 0.0        # 바깥 2px 띠의 밝은 픽셀 비율
    white: float = 0.0       # 딱 붙은 1px 띠의 흰색 픽셀 비율

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class IconRule:
    """미니맵 아이콘 하나를 찾는 규칙 (OpenCV HSV 기준: H 0~179, S/V 0~255)."""

    h_min: int = 135
    h_max: int = 155
    s_min: int = 90
    s_max: int = 255
    v_min: int = 150
    v_max: int = 255

    min_area: int = 18           # 덩어리 최소 픽셀 수
    max_area: int = 250
    min_side: int = 4            # 바운딩 박스 변 길이
    max_side: int = 22
    max_aspect: float = 1.6      # 긴 변 / 짧은 변 허용치 (1.0 이면 정사각형만)

    fill_min: float = 0.40
    fill_max: float = 1.0

    border_ratio: float = 0.15   # 밝은 테두리 비율 하한 (0이면 검사 안 함)
    border_s_max: int = 100      # '밝다' 로 볼 채도 상한
    border_v_min: int = 140      # '밝다' 로 볼 밝기 하한

    white_ratio: float = 0.12    # 흰 테두리 비율 하한 (0이면 검사 안 함)
    white_s_max: int = 60        # '흰색' 으로 볼 채도 상한
    white_v_min: int = 185       # '흰색' 으로 볼 밝기 하한

    min_count: int = 1           # 이 개수 이상 찾으면 조건 성립

    # 마름모 테두리 훑기 (모듈 설명 참고). ring_scan 이 0 이면 하지 않는다.
    ring_scan: float = 0.0       # 네 변 중 가장 어두운 변의 밝은 픽셀 비율 하한
    ring_mean: float = 0.4       # 네 변 평균 하한
    ring_s_max: int = 90         # 테두리로 볼 채도 상한 (압축된 회색 테두리까지)
    ring_v_min: int = 120        # 테두리로 볼 밝기 하한
    ring_r_min: int = 2          # 훑어 볼 마름모 반지름(px) 범위
    ring_r_max: int = 10

    # ------------------------------------------------------------------ #
    def color_mask(self, hsv: np.ndarray) -> np.ndarray:
        lo_s, hi_s = int(self.s_min), int(self.s_max)
        lo_v, hi_v = int(self.v_min), int(self.v_max)
        if self.h_min <= self.h_max:
            mask = cv2.inRange(hsv, (int(self.h_min), lo_s, lo_v),
                                    (int(self.h_max), hi_s, hi_v))
        else:
            # 빨강처럼 0/179 를 넘나드는 색상 범위
            m1 = cv2.inRange(hsv, (int(self.h_min), lo_s, lo_v), (179, hi_s, hi_v))
            m2 = cv2.inRange(hsv, (0, lo_s, lo_v), (int(self.h_max), hi_s, hi_v))
            mask = cv2.bitwise_or(m1, m2)
        return mask

    def build_mask(self, bgr: np.ndarray) -> np.ndarray:
        mask = self.color_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        found = self._detect_blobs(hsv)
        if self.ring_scan > 0:
            for d in self._scan_rings(hsv):
                cx, cy = d.center
                if all(max(abs(cx - f.center[0]), abs(cy - f.center[1])) > 4
                       for f in found):
                    found.append(d)
        return found

    def _detect_blobs(self, hsv: np.ndarray) -> List[Detection]:
        mask = cv2.morphologyEx(self.color_mask(hsv), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        S, V = hsv[..., 1], hsv[..., 2]
        bright = (S < self.border_s_max) & (V > self.border_v_min)
        white = (S < self.white_s_max) & (V > self.white_v_min)

        img_h, img_w = mask.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        found: List[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not (self.min_area <= area <= self.max_area):
                continue
            if not (self.min_side <= w <= self.max_side):
                continue
            if not (self.min_side <= h <= self.max_side):
                continue
            if max(w, h) / float(min(w, h)) > self.max_aspect:
                continue
            fill = area / float(w * h)
            if not (self.fill_min <= fill <= self.fill_max):
                continue

            ring = self._ring_ratio(bright, x, y, w, h, img_w, img_h)
            if ring < self.border_ratio:
                continue
            wr = self._white_ratio(white, labels, i, x, y, w, h, img_w, img_h)
            if wr < self.white_ratio:
                continue

            found.append(Detection(x, y, w, h, round(fill, 3),
                                   round(ring, 3), round(wr, 3)))
        return found

    def _scan_rings(self, hsv: np.ndarray) -> List[Detection]:
        """보라 픽셀마다 '여기가 룬 한가운데라면' 하고 마름모 테두리를 재 본다.

        반지름 r 마다
          core  중심에서 r-2 안쪽 마름모가 아이콘 색으로 절반 이상 차 있는가
          ring  r~r+1 마름모 띠의 네 변에서 각각 밝은(테두리) 픽셀 비율
        네 변 중 가장 낮은 값이 가장 큰 r 을 그 점의 점수로 삼고,
        5x5 안에서 가장 높은 점만 남긴다.
        """
        color = (self.color_mask(hsv) > 0).astype(np.float32)
        S, V = hsv[..., 1], hsv[..., 2]
        light = ((S < self.ring_s_max) & (V > self.ring_v_min)).astype(np.float32)
        light *= 1.0 - color        # 아이콘 색 자체는 테두리로 치지 않는다

        best = np.zeros(color.shape, np.float32)
        best_mean = np.zeros_like(best)
        best_r = np.zeros(color.shape, np.int32)
        for r in range(max(1, int(self.ring_r_min)), int(self.ring_r_max) + 1):
            k_core, k_sides = _diamond_kernels(r)
            core = _filt(color, k_core)
            sides = np.stack([_filt(light, k) for k in k_sides])
            ok = (core >= 0.5) & (color > 0)
            lo = np.where(ok, sides.min(0), 0)
            better = lo > best
            best[better] = lo[better]
            best_mean[better] = sides.mean(0)[better]
            best_r[better] = r

        peak = best >= cv2.dilate(best, np.ones((5, 5), np.uint8))
        ys, xs = np.where(peak & (best >= self.ring_scan) &
                          (best_mean >= self.ring_mean))
        out: List[Detection] = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            r = int(best_r[y, x])
            out.append(Detection(x - r, y - r, 2 * r + 1, 2 * r + 1, 0.0,
                                 round(float(best_mean[y, x]), 3),
                                 round(float(best[y, x]), 3)))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ring_ratio(bright, x, y, w, h, img_w, img_h) -> float:
        """바운딩 박스 바깥 2px 띠에서 밝은 픽셀이 차지하는 비율."""
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(img_w, x + w + 2), min(img_h, y + h + 2)
        sub = bright[y0:y1, x0:x1]
        inner = np.zeros(sub.shape, bool)
        inner[(y - y0):(y - y0 + h), (x - x0):(x - x0 + w)] = True
        band = ~inner
        n = int(band.sum())
        return float((sub & band).sum()) / n if n else 0.0

    @staticmethod
    def _white_ratio(white, labels, label, x, y, w, h, img_w, img_h) -> float:
        """덩어리 모양을 1px 부풀린 띠에서 흰 픽셀이 차지하는 비율.

        박스가 아니라 **덩어리 모양**을 따라가는 게 중요하다. 마름모는 박스
        네 귀퉁이가 배경이라, 박스 기준으로 재면 테두리가 묻혀 버린다.
        """
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(img_w, x + w + 2), min(img_h, y + h + 2)
        comp = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        grown = cv2.dilate(comp, np.ones((3, 3), np.uint8))
        band = (grown > 0) & (comp == 0)
        n = int(band.sum())
        return float(white[y0:y1, x0:x1][band].sum()) / n if n else 0.0


@lru_cache(maxsize=None)
def _diamond_kernels(r: int):
    """반지름 r 마름모의 안쪽(core) 커널과, r~r+1 띠를 네 변으로 나눈 커널 4개."""
    R = r + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.abs(xx) + np.abs(yy)
    band = (d >= r) & (d <= r + 1)
    core = (d <= max(0, r - 2)).astype(np.float32)
    sides = [band & (yy < 0) & (xx >= 0), band & (yy <= 0) & (xx < 0),
             band & (yy > 0) & (xx <= 0), band & (yy >= 0) & (xx > 0)]
    norm = lambda k: k.astype(np.float32) / float(k.sum())
    return norm(core), tuple(norm(k) for k in sides)


def _filt(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # 화면 밖은 0 으로 본다 (가장자리 아이콘이 테두리를 '빌려 오지' 않도록)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)


def draw_boxes(bgr: np.ndarray, dets: List[Detection],
               color=(0, 255, 0)) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        cv2.rectangle(out, (d.x - 2, d.y - 2), (d.x + d.w + 1, d.y + d.h + 1),
                      color, 1)
    return out
