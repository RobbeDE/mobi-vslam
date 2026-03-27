import numpy as np
import heapq
import numpy as np

class RiskAStar:
    def __init__(self, risk_map, threshold=0.7, risk_weight=5.0, step_size=1.0):
        self.risk_map = risk_map
        self.threshold = threshold
        self.risk_weight = risk_weight
        self.step_size = step_size

    def plan(self, current_pose, goal_pose):
        start = np.array(current_pose, dtype=float)
        goal = np.array(goal_pose, dtype=float)

        print("Planning path from", start, "to", goal)

        raw_path = self._astar(start, goal)
        if raw_path is None or len(raw_path) < 2:
            return[]

        # Return the FULL resampled path for the hybrid tracker
        # waypoints = resample_path(raw_path, step_size=self.step_size)
        return raw_path

    def _astar(self, start, goal):
        rows, cols = self.risk_map.shape
        sx = int(np.clip(round(start[0]), 0, rows - 1))
        sy = int(np.clip(round(start[1]), 0, cols - 1))
        gx = int(np.clip(round(goal[0]), 0, rows - 1))
        gy = int(np.clip(round(goal[1]), 0, cols - 1))

        neighbours =[(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]

        open_set = [(0.0, sx, sy)]
        came_from = {}
        g_score = {(sx, sy): 0.0}
        closed = set()

        while open_set:
            f, x, y = heapq.heappop(open_set)
            if (x, y) in closed:
                continue
            closed.add((x, y))

            if x == gx and y == gy:
                path =[]
                cx, cy = gx, gy
                while (cx, cy) in came_from:
                    path.append(np.array([float(cx), float(cy)]))
                    cx, cy = came_from[(cx, cy)]
                path.append(np.array([float(sx), float(sy)]))
                return np.array(path[::-1])

            for dx, dy in neighbours:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue
                if (nx, ny) in closed:
                    continue
                risk = float(self.risk_map[ny, nx]) # IMPORTANT: INDEXING IN NP ARRAYS IS [ROW, COL] = [Y, X]
                if risk >= self.threshold:
                    print(f"Skipping ({nx}, {ny}) due to high risk: {risk:.2f}")
                    continue
                step_d = np.sqrt(dx * dx + dy * dy)
                edge_cost = step_d * (1.0 + self.risk_weight * risk)
                tentative = g_score[(x, y)] + edge_cost

                if (nx, ny) not in g_score or tentative < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative
                    h = np.sqrt((nx - gx) ** 2 + (ny - gy) ** 2)
                    heapq.heappush(open_set, (tentative + h, nx, ny))
                    came_from[(nx, ny)] = (x, y)
        
        return None