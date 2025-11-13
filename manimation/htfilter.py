from manim import *
import numpy as np

class HierarchicalMedianFilter(Scene):
    def construct(self):
        # Title
        title = Text("Hierarchical Tiling Median Filter", font_size=42)
        subtitle = Text("Fast Parallel Algorithm (Sugy 2025)", font_size=28, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        self.play(Write(title), FadeIn(subtitle))
        self.wait()
        
        # Create a 4x4 tile with 5x5 kernel example
        tile_size = 4
        kernel_size = 5
        
        # Show the concept
        concept_text = Text("Key Idea: Process multiple pixels together", font_size=32)
        concept_text.next_to(title_group, DOWN, buff=0.5)
        self.play(Write(concept_text))
        self.wait()
        
        # Create sample 8x8 image region with noise
        image_data = np.array([
            [100, 110, 105, 255, 102, 108, 105, 110],
            [105, 108, 110, 107, 105, 255, 108, 105],
            [102, 108, 0,   105, 110, 107, 105, 108],
            [108, 105, 107, 110, 108, 105, 255, 110],
            [100, 103, 105, 108, 102, 110, 108, 105],
            [105, 110, 108, 0,   105, 107, 105, 108],
            [108, 105, 107, 110, 108, 105, 110, 102],
            [102, 108, 105, 108, 255, 110, 108, 105]
        ])
        
        self.play(FadeOut(concept_text))
        self.wait(0.3)
        
        # Show 4x4 root tile
        self.show_root_tile(image_data, tile_size, kernel_size, title_group)
        self.wait(1)
        
        # Show hierarchical splitting
        self.show_hierarchical_split(title_group)
        self.wait(1)
        
        # Show complexity comparison
        self.show_complexity_comparison(title_group)
        self.wait(2)
    
    def show_root_tile(self, image_data, tile_size, kernel_size, title_group):
        label = Text("4×4 Root Tile with 5×5 Kernels", font_size=32)
        label.next_to(title_group, DOWN, buff=0.3)
        self.play(Write(label))
        
        # Create grid
        grid = self.create_image_grid(image_data, 0.45, show_values=False)
        grid.next_to(label, DOWN, buff=0.4)
        self.play(FadeIn(grid))
        
        # Highlight 4x4 tile in center
        tile_squares = []
        for i in range(2, 6):
            for j in range(2, 6):
                idx = i * 8 + j
                tile_squares.append(grid[idx])
        
        tile_group = VGroup(*tile_squares)
        tile_box = SurroundingRectangle(tile_group, color=BLUE, buff=0.05, stroke_width=4)
        tile_label = Text("4×4 Tile", font_size=24, color=BLUE)
        tile_label.next_to(tile_box, RIGHT, buff=0.3)
        
        self.play(Create(tile_box), Write(tile_label))
        self.wait(0.5)
        
        # Show footprint (union of all kernels)
        footprint_squares = []
        for i in range(0, 8):
            for j in range(0, 8):
                idx = i * 8 + j
                footprint_squares.append(grid[idx])
        
        footprint_group = VGroup(*footprint_squares)
        footprint_box = SurroundingRectangle(footprint_group, color=YELLOW, buff=0.05, stroke_width=3)
        footprint_label = Text("Footprint: 8×8", font_size=24, color=YELLOW)
        footprint_label.next_to(footprint_box, LEFT, buff=0.3)
        
        self.play(Create(footprint_box), Write(footprint_label))
        self.wait(0.5)
        
        # Show core (intersection of all kernels)
        core_squares = []
        for i in range(3, 5):
            for j in range(3, 5):
                idx = i * 8 + j
                square = grid[idx].copy()
                square.set_fill(GREEN, opacity=0.3)
                core_squares.append(square)
        
        core_label = Text("Core: 2×2\n(shared by all)", font_size=24, color=GREEN)
        core_label.next_to(grid, DOWN, buff=0.5)
        
        self.play(
            *[FadeIn(sq) for sq in core_squares],
            Write(core_label)
        )
        self.wait(1)
        
        # Clean up
        self.play(
            FadeOut(grid), FadeOut(tile_box), FadeOut(tile_label),
            FadeOut(footprint_box), FadeOut(footprint_label),
            *[FadeOut(sq) for sq in core_squares],
            FadeOut(core_label), FadeOut(label)
        )
    
    def show_hierarchical_split(self, title_group):
        split_title = Text("Hierarchical Splitting Strategy", font_size=32)
        split_title.next_to(title_group, DOWN, buff=0.5)
        self.play(Write(split_title))
        
        # Create tree diagram
        root = Square(side_length=1.5, color=BLUE, fill_opacity=0.3)
        root_label = Text("4×4", font_size=20).move_to(root)
        root_node = VGroup(root, root_label)
        root_node.shift(UP * 0.5)
        
        self.play(Create(root), Write(root_label))
        self.wait(0.3)
        
        # First split: 4x4 -> two 2x4
        left_1 = Square(side_length=1.2, color=GREEN, fill_opacity=0.3)
        left_1_label = Text("2×4", font_size=18).move_to(left_1)
        left_1_node = VGroup(left_1, left_1_label)
        left_1_node.shift(DOWN * 1 + LEFT * 2)
        
        right_1 = left_1_node.copy()
        right_1.shift(RIGHT * 4)
        
        arrow_l1 = Arrow(root.get_bottom(), left_1.get_top(), buff=0.1, stroke_width=2)
        arrow_r1 = Arrow(root.get_bottom(), right_1[0].get_top(), buff=0.1, stroke_width=2)
        
        self.play(
            Create(arrow_l1), Create(arrow_r1),
            Create(left_1), Write(left_1_label),
            Create(right_1[0]), Write(right_1[1])
        )
        self.wait(0.3)
        
        # Second split: 2x4 -> two 2x2
        ll_2 = Square(side_length=0.9, color=YELLOW, fill_opacity=0.3)
        ll_2_label = Text("2×2", font_size=16).move_to(ll_2)
        ll_2_node = VGroup(ll_2, ll_2_label)
        ll_2_node.shift(DOWN * 2.3 + LEFT * 3)
        
        lr_2 = ll_2_node.copy().shift(RIGHT * 1.5)
        rl_2 = ll_2_node.copy().shift(RIGHT * 3)
        rr_2 = ll_2_node.copy().shift(RIGHT * 4.5)
        
        arrows_2 = [
            Arrow(left_1.get_bottom(), ll_2.get_top(), buff=0.05, stroke_width=2),
            Arrow(left_1.get_bottom(), lr_2[0].get_top(), buff=0.05, stroke_width=2),
            Arrow(right_1[0].get_bottom(), rl_2[0].get_top(), buff=0.05, stroke_width=2),
            Arrow(right_1[0].get_bottom(), rr_2[0].get_top(), buff=0.05, stroke_width=2),
        ]
        
        self.play(
            *[Create(arr) for arr in arrows_2],
            Create(ll_2), Write(ll_2_label),
            Create(lr_2[0]), Write(lr_2[1]),
            Create(rl_2[0]), Write(rl_2[1]),
            Create(rr_2[0]), Write(rr_2[1])
        )
        self.wait(0.3)
        
        # Continue to 1x1 leaves
        leaves = []
        arrows_3 = []
        for i, parent in enumerate([ll_2, lr_2[0], rl_2[0], rr_2[0]]):
            for j in range(2):
                leaf = Square(side_length=0.5, color=RED, fill_opacity=0.3)
                leaf_label = Text("1×1", font_size=12).move_to(leaf)
                leaf_node = VGroup(leaf, leaf_label)
                x_pos = -4.5 + i * 1.5 + j * 0.6
                leaf_node.shift(DOWN * 3.5 + RIGHT * x_pos)
                leaves.append(leaf_node)
                arrows_3.append(Arrow(parent.get_bottom(), leaf.get_top(), buff=0.05, stroke_width=1.5))
        
        self.play(
            *[Create(arr) for arr in arrows_3],
            *[Create(leaf[0]) for leaf in leaves],
            *[Write(leaf[1]) for leaf in leaves]
        )
        
        explanation = Text("Each level shares work between pixels", font_size=24, color=BLUE)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(1)
        
        # Clean up
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != title_group[0] and mob != title_group[1]])
    
    def show_complexity_comparison(self, title_group):
        comp_title = Text("Computational Complexity", font_size=36)
        comp_title.next_to(title_group, DOWN, buff=0.5)
        self.play(Write(comp_title))
        
        # Create comparison table
        methods = [
            ("Naive (per-pixel sort)", "O(k²)", RED),
            ("Traditional sorting network", "O(k² log²k)", ORANGE),
            ("Hierarchical (data-oblivious)", "O(k log k)", YELLOW),
            ("Hierarchical (data-aware)", "O(k)", GREEN),
        ]
        
        table_group = VGroup()
        for i, (method, complexity, color) in enumerate(methods):
            method_text = Text(method, font_size=24).shift(LEFT * 2)
            complexity_text = Text(complexity, font_size=28, color=color).shift(RIGHT * 2.5)
            row = VGroup(method_text, complexity_text)
            row.shift(DOWN * (0.8 * i))
            table_group.add(row)
        
        table_group.next_to(comp_title, DOWN, buff=0.7)
        
        for row in table_group:
            self.play(Write(row[0]), Write(row[1]), run_time=0.5)
        self.wait(0.5)
        
        # Highlight the improvement
        best_box = SurroundingRectangle(table_group[-1], color=GREEN, buff=0.1, stroke_width=3)
        best_label = Text("Asymptotically better!", font_size=28, color=GREEN)
        best_label.next_to(best_box, DOWN, buff=0.3)
        
        self.play(Create(best_box), Write(best_label))
        self.wait(0.5)
        
        # Performance note
        perf_text = Text("Up to 5× faster on GPU vs. state-of-the-art", font_size=26, color=BLUE)
        perf_text.to_edge(DOWN)
        self.play(Write(perf_text))
        self.wait(1)
    
    def create_image_grid(self, data, cell_size, show_values=True):
        rows, cols = data.shape
        grid = VGroup()
        
        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size, stroke_width=1.5)
                
                intensity = data[i, j] / 255
                if data[i, j] == 255 or data[i, j] == 0:
                    cell.set_fill(RED if data[i, j] == 255 else BLUE, opacity=0.7)
                else:
                    cell.set_fill(WHITE, opacity=intensity * 0.5)
                
                if show_values and (data[i, j] == 255 or data[i, j] == 0):
                    value_text = Text(str(data[i, j]), font_size=14, color=WHITE)
                    value_text.move_to(cell.get_center())
                    cell.add(value_text)
                
                cell.move_to(np.array([j * cell_size, -i * cell_size, 0]))
                grid.add(cell)
        
        return grid