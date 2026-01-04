import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


class SocialNetworkPropagation:
    def __init__(self, n_nodes=100, edge_probability=0.05):
        """Khởi tạo mạng xã hội"""
        self.G = nx.erdos_renyi_graph(n_nodes, edge_probability)
        self.n_nodes = n_nodes

        # Gán nội dung (content) cho mỗi node (0-4 categories)
        self.node_content = {node: random.randint(0, 4) for node in self.G.nodes()}

    def random_walk(self, start_node, steps=100):
        """Random Walk thường - chọn ngẫu nhiên láng giềng"""
        visited = set()
        current = start_node
        path = [current]
        visited.add(current)

        coverage_over_time = [1]  # Số node đã thăm theo thời gian

        for _ in range(steps):
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            path.append(current)
            visited.add(current)
            coverage_over_time.append(len(visited))

        return path, visited, coverage_over_time

    def content_biased_random_walk(self, start_node, steps=100, bias=3.0):
        """
        Content-biased Random Walk - ưu tiên láng giềng có nội dung giống
        bias: hệ số ưu tiên (càng cao càng thiên vị nội dung giống)
        """
        visited = set()
        current = start_node
        path = [current]
        visited.add(current)

        coverage_over_time = [1]

        for _ in range(steps):
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break

            # Tính xác suất dựa trên nội dung
            current_content = self.node_content[current]
            weights = []

            for neighbor in neighbors:
                if self.node_content[neighbor] == current_content:
                    weights.append(bias)  # Nội dung giống -> trọng số cao
                else:
                    weights.append(1.0)  # Nội dung khác -> trọng số thấp

            # Normalize weights
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

            # Chọn láng giềng theo xác suất
            current = random.choices(neighbors, weights=probabilities)[0]
            path.append(current)
            visited.add(current)
            coverage_over_time.append(len(visited))

        return path, visited, coverage_over_time

    def compare_propagation(self, start_node=0, steps=100, n_simulations=50):
        """So sánh Random Walk và Content-biased Random Walk"""

        # Chạy nhiều lần simulation
        rw_coverages = []
        cbrw_coverages = []

        for _ in range(n_simulations):
            # Random Walk
            _, _, rw_cov = self.random_walk(start_node, steps)
            rw_coverages.append(rw_cov)

            # Content-biased Random Walk
            _, _, cbrw_cov = self.content_biased_random_walk(start_node, steps)
            cbrw_coverages.append(cbrw_cov)

        # Tính trung bình
        rw_avg = np.mean(rw_coverages, axis=0)
        cbrw_avg = np.mean(cbrw_coverages, axis=0)

        return rw_avg, cbrw_avg, rw_coverages, cbrw_coverages

    def visualize_network(self, path=None, title="Social Network"):
        """Vẽ mạng xã hội"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.G, seed=42)

        # Màu sắc theo content
        node_colors = [self.node_content[node] for node in self.G.nodes()]

        # Vẽ nodes
        nodes = nx.draw_networkx_nodes(
            self.G, pos, node_color=node_colors, node_size=300, cmap="Set3", alpha=0.8
        )

        # Vẽ edges
        nx.draw_networkx_edges(self.G, pos, alpha=0.2)

        # Highlight path nếu có
        if path:
            path_edges = [
                (path[i], path[i + 1])
                for i in range(len(path) - 1)
                if self.G.has_edge(path[i], path[i + 1])
            ]
            nx.draw_networkx_edges(
                self.G, pos, path_edges, edge_color="red", width=2, alpha=0.6
            )

            # Highlight start node
            nx.draw_networkx_nodes(
                self.G,
                pos,
                [path[0]],
                node_color="red",
                node_size=500,
                node_shape="s",
                label="Start",
            )

        plt.title(title, fontsize=16, fontweight="bold")
        plt.colorbar(nodes, label="Content Category", shrink=0.8)
        plt.axis("off")
        plt.tight_layout()

    def visualize_comparison(self, rw_avg, cbrw_avg):
        """Vẽ so sánh tốc độ lan truyền và độ phủ"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        steps = range(len(rw_avg))

        # Plot 1: Coverage over time
        axes[0].plot(
            steps,
            rw_avg,
            "b-",
            linewidth=2,
            label="Random Walk",
            marker="o",
            markersize=3,
        )
        axes[0].plot(
            steps,
            cbrw_avg,
            "r-",
            linewidth=2,
            label="Content-biased Random Walk",
            marker="s",
            markersize=3,
        )
        axes[0].set_xlabel("Steps", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Coverage (Visited Nodes)", fontsize=12, fontweight="bold")
        axes[0].set_title(
            "Coverage Over Time Comparison", fontsize=14, fontweight="bold"
        )
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Propagation speed (derivative)
        rw_speed = np.diff(rw_avg)
        cbrw_speed = np.diff(cbrw_avg)

        axes[1].plot(
            steps[1:],
            rw_speed,
            "b-",
            linewidth=2,
            label="Random Walk",
            marker="o",
            markersize=3,
        )
        axes[1].plot(
            steps[1:],
            cbrw_speed,
            "r-",
            linewidth=2,
            label="Content-biased Random Walk",
            marker="s",
            markersize=3,
        )
        axes[1].set_xlabel("Steps", fontsize=12, fontweight="bold")
        axes[1].set_ylabel(
            "Propagation Speed (nodes/step)", fontsize=12, fontweight="bold"
        )
        axes[1].set_title(
            "Propagation Speed Comparison", fontsize=14, fontweight="bold"
        )
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

    def visualize_final_comparison(self, rw_avg, cbrw_avg):
        """Biểu đồ tổng hợp cuối cùng"""
        fig = plt.figure(figsize=(16, 10))

        # Subplot 1: Coverage comparison
        ax1 = plt.subplot(2, 2, 1)
        steps = range(len(rw_avg))
        ax1.plot(steps, rw_avg, "b-", linewidth=2.5, label="Random Walk")
        ax1.plot(steps, cbrw_avg, "r-", linewidth=2.5, label="Content-biased RW")
        ax1.fill_between(steps, rw_avg, alpha=0.3, color="blue")
        ax1.fill_between(steps, cbrw_avg, alpha=0.3, color="red")
        ax1.set_xlabel("Steps", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Coverage (Nodes)", fontsize=11, fontweight="bold")
        ax1.set_title("Coverage Over Time", fontsize=13, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Final coverage bar chart
        ax2 = plt.subplot(2, 2, 2)
        final_coverages = [rw_avg[-1], cbrw_avg[-1]]
        bars = ax2.bar(
            ["Random Walk", "Content-biased RW"],
            final_coverages,
            color=["blue", "red"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax2.set_ylabel("Visited Nodes", fontsize=11, fontweight="bold")
        ax2.set_title("Final Coverage", fontsize=13, fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Subplot 3: Speed comparison
        ax3 = plt.subplot(2, 2, 3)
        rw_speed = np.diff(rw_avg)
        cbrw_speed = np.diff(cbrw_avg)
        ax3.plot(steps[1:], rw_speed, "b-", linewidth=2, label="Random Walk", alpha=0.8)
        ax3.plot(
            steps[1:],
            cbrw_speed,
            "r-",
            linewidth=2,
            label="Content-biased RW",
            alpha=0.8,
        )
        ax3.set_xlabel("Steps", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Speed (nodes/step)", fontsize=11, fontweight="bold")
        ax3.set_title("Propagation Speed", fontsize=13, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Subplot 4: Statistics table
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis("off")

        # Calculate statistics
        stats_data = [
            ["Metric", "Random Walk", "Content-biased RW"],
            ["Final Coverage", f"{rw_avg[-1]:.1f}", f"{cbrw_avg[-1]:.1f}"],
            ["Avg Speed", f"{np.mean(rw_speed):.3f}", f"{np.mean(cbrw_speed):.3f}"],
            ["Max Speed", f"{np.max(rw_speed):.3f}", f"{np.max(cbrw_speed):.3f}"],
            ["Efficiency (%)", "100%", f"{(cbrw_avg[-1] / rw_avg[-1] * 100):.1f}%"],
        ]

        table = ax4.table(
            cellText=stats_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.35, 0.3, 0.35],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Style data rows
        for i in range(1, len(stats_data)):
            for j in range(3):
                if j == 0:
                    table[(i, j)].set_facecolor("#E8F5E9")
                    table[(i, j)].set_text_props(weight="bold")
                else:
                    table[(i, j)].set_facecolor("#F5F5F5")

        ax4.set_title("Comparison Statistics", fontsize=13, fontweight="bold", pad=20)

        plt.suptitle(
            "Social Network Propagation: Random Walk vs Content-biased Random Walk",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()


def main():
    print("=" * 70)
    print("  PHÂN TÍCH LAN TRUYỀN MẠNG XÃ HỘI")
    print("  Random Walk vs Content-biased Random Walk")
    print("=" * 70)

    # Khởi tạo mạng
    print("\n[1] Đang tạo mạng xã hội...")
    network = SocialNetworkPropagation(n_nodes=100, edge_probability=0.05)
    print(
        f"    ✓ Đã tạo mạng với {network.G.number_of_nodes()} nodes và {network.G.number_of_edges()} edges"
    )

    # Chạy simulations
    print("\n[2] Đang chạy simulations (50 lần)...")
    start_node = 0
    rw_avg, cbrw_avg, rw_all, cbrw_all = network.compare_propagation(
        start_node=start_node, steps=100, n_simulations=50
    )
    print("    ✓ Hoàn thành simulations")

    # In kết quả
    print("\n[3] KẾT QUẢ:")
    print(f"    Random Walk - Độ phủ cuối: {rw_avg[-1]:.1f} nodes")
    print(f"    Content-biased RW - Độ phủ cuối: {cbrw_avg[-1]:.1f} nodes")
    print(f"    Chênh lệch: {abs(rw_avg[-1] - cbrw_avg[-1]):.1f} nodes")
    print(f"    Tốc độ TB - Random Walk: {np.mean(np.diff(rw_avg)):.3f} nodes/bước")
    print(
        f"    Tốc độ TB - Content-biased: {np.mean(np.diff(cbrw_avg)):.3f} nodes/bước"
    )

    # Tạo visualizations
    print("\n[4] Đang tạo biểu đồ...")

    # 1. Mạng cơ bản (không có path)
    plt.figure()
    network.visualize_network(None, "Social Network - Structure")
    plt.savefig("data/01_network_structure.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [1] Đã lưu: 01_network_structure.png")

    # 2. Random Walk path
    path_rw, _, _ = network.random_walk(start_node, steps=30)
    plt.figure()
    network.visualize_network(path_rw, "Random Walk - Sample Path")
    plt.savefig("data/02_random_walk_path.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [2] Đã lưu: 02_random_walk_path.png")

    # 3. Content-biased Random Walk path
    path_cbrw, _, _ = network.content_biased_random_walk(start_node, steps=30)
    plt.figure()
    network.visualize_network(path_cbrw, "Content-biased Random Walk - Sample Path")
    plt.savefig("data/03_content_biased_path.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [3] Đã lưu: 03_content_biased_path.png")

    # 4. So sánh độ phủ theo thời gian (riêng)
    plt.figure(figsize=(10, 6))
    steps = range(len(rw_avg))
    plt.plot(
        steps,
        rw_avg,
        "b-",
        linewidth=3,
        label="Random Walk",
        marker="o",
        markersize=4,
        markevery=5,
    )
    plt.plot(
        steps,
        cbrw_avg,
        "r-",
        linewidth=3,
        label="Content-biased RW",
        marker="s",
        markersize=4,
        markevery=5,
    )
    plt.fill_between(steps, rw_avg, alpha=0.2, color="blue")
    plt.fill_between(steps, cbrw_avg, alpha=0.2, color="red")
    plt.xlabel("Steps", fontsize=13, fontweight="bold")
    plt.ylabel("Coverage (Visited Nodes)", fontsize=13, fontweight="bold")
    plt.title("Coverage Over Time Comparison", fontsize=15, fontweight="bold")
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("data/04_coverage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [4] Đã lưu: 04_coverage_comparison.png")

    # 5. So sánh tốc độ lan truyền (riêng)
    plt.figure(figsize=(10, 6))
    rw_speed = np.diff(rw_avg)
    cbrw_speed = np.diff(cbrw_avg)
    plt.plot(
        steps[1:],
        rw_speed,
        "b-",
        linewidth=3,
        label="Random Walk",
        marker="o",
        markersize=4,
        markevery=5,
        alpha=0.8,
    )
    plt.plot(
        steps[1:],
        cbrw_speed,
        "r-",
        linewidth=3,
        label="Content-biased RW",
        marker="s",
        markersize=4,
        markevery=5,
        alpha=0.8,
    )
    plt.xlabel("Steps", fontsize=13, fontweight="bold")
    plt.ylabel("Propagation Speed (nodes/step)", fontsize=13, fontweight="bold")
    plt.title("Propagation Speed Comparison", fontsize=15, fontweight="bold")
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.axhline(
        y=np.mean(rw_speed),
        color="blue",
        linestyle=":",
        alpha=0.5,
        label=f"TB RW: {np.mean(rw_speed):.3f}",
    )
    plt.axhline(
        y=np.mean(cbrw_speed),
        color="red",
        linestyle=":",
        alpha=0.5,
        label=f"TB CBRW: {np.mean(cbrw_speed):.3f}",
    )
    plt.tight_layout()
    plt.savefig("data/05_speed_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [5] Đã lưu: 05_speed_comparison.png")

    # 6. Biểu đồ cột so sánh độ phủ cuối
    plt.figure(figsize=(8, 6))
    categories = ["Random Walk", "Content-biased\nRandom Walk"]
    values = [rw_avg[-1], cbrw_avg[-1]]
    colors = ["#2196F3", "#F44336"]
    bars = plt.bar(
        categories,
        values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
        width=0.6,
    )
    plt.ylabel("Visited Nodes", fontsize=13, fontweight="bold")
    plt.title("Final Coverage Comparison", fontsize=15, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.ylim(0, max(values) * 1.2)

    # Thêm giá trị lên cột
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig("data/06_final_coverage_bars.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [6] Đã lưu: 06_final_coverage_bars.png")

    # 7. So sánh 2 phương pháp (2 panels)
    network.visualize_comparison(rw_avg, cbrw_avg)
    plt.savefig("data/07_comparison_dual_panel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [7] Đã lưu: 07_comparison_dual_panel.png")

    # 8. Biểu đồ tổng hợp đầy đủ
    network.visualize_final_comparison(rw_avg, cbrw_avg)
    plt.savefig("data/08_full_comparison_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✓ [8] Đã lưu: 08_full_comparison_dashboard.png")

    print("\n[5] Hiển thị biểu đồ cuối...")
    # Hiển thị lại biểu đồ tổng hợp
    network.visualize_final_comparison(rw_avg, cbrw_avg)
    plt.show()

    print("\n" + "=" * 70)
    print("  HOÀN THÀNH!")
    print("=" * 70)


if __name__ == "__main__":
    main()
