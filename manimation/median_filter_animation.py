from manim import *
import numpy as np

class MedianFilter(Scene):
    def construct(self):
        # Title
        title = Text("3×3 Median Filter Operation", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create a sample 5x5 image with noise
        image_data = np.array([
            [100, 110, 105, 108, 102],
            [105, 255, 110, 107, 105],  # 255 is noise (salt)
            [102, 108, 0,   105, 110],  # 0 is noise (pepper)
            [108, 105, 107, 255, 108],  # 255 is noise
            [100, 103, 105, 108, 102]
        ])
        
        # Create input image grid
        input_label = Text("Input Image (with noise)", font_size=32)
        input_label.next_to(title, DOWN, buff=0.5).shift(LEFT * 3.5)
        
        input_grid = self.create_image_grid(image_data, 0.6)
        input_grid.next_to(input_label, DOWN, buff=0.3)
        
        self.play(FadeIn(input_label), FadeIn(input_grid))
        self.wait()
        
        # Create output image grid (initially empty)
        output_label = Text("Output Image (filtered)", font_size=32)
        output_label.next_to(title, DOWN, buff=0.5).shift(RIGHT * 3.5)
        
        output_data = np.copy(image_data)
        output_grid = self.create_image_grid(output_data, 0.6, show_values=False)
        output_grid.next_to(output_label, DOWN, buff=0.3)
        
        self.play(FadeIn(output_label), FadeIn(output_grid))
        self.wait()
        
        # Process center pixel (2,2)
        self.process_pixel(image_data, output_data, 2, 2, input_grid, output_grid)
        self.wait(0.5)
        
        # Process pixel (1,2)
        self.process_pixel(image_data, output_data, 1, 2, input_grid, output_grid)
        self.wait(0.5)
        
        # Process pixel (2,3)
        self.process_pixel(image_data, output_data, 2, 3, input_grid, output_grid)
        self.wait(0.5)
        
        # Final message
        final_text = Text("Median filter removes salt & pepper noise!", font_size=36, color=GREEN)
        final_text.to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)
    
    def create_image_grid(self, data, cell_size, show_values=True):
        rows, cols = data.shape
        grid = VGroup()
        
        for i in range(rows):
            for j in range(cols):
                # Create cell
                cell = Square(side_length=cell_size, stroke_width=2)
                
                # Color based on intensity
                intensity = data[i, j] / 255
                if show_values:
                    if data[i, j] == 255 or data[i, j] == 0:
                        cell.set_fill(RED if data[i, j] == 255 else BLUE, opacity=0.7)
                    else:
                        cell.set_fill(WHITE, opacity=intensity * 0.5)
                else:
                    cell.set_fill(GRAY, opacity=0.2)
                
                # Add value text
                if show_values:
                    value_text = Text(str(data[i, j]), font_size=18)
                    value_text.move_to(cell.get_center())
                    cell.add(value_text)
                
                # Position cell
                cell.move_to(np.array([j * cell_size, -i * cell_size, 0]))
                grid.add(cell)
        
        return grid
    
    def process_pixel(self, input_data, output_data, row, col, input_grid, output_grid):
        # Highlight the 3x3 window
        window_squares = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                idx = (row + i) * 5 + (col + j)
                square = input_grid[idx].copy()
                window_squares.append(square)
        
        window_group = VGroup(*window_squares)
        highlight = SurroundingRectangle(window_group, color=YELLOW, buff=0.05, stroke_width=4)
        
        self.play(Create(highlight))
        
        # Extract 3x3 window values
        window_values = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                window_values.append(input_data[row + i, col + j])
        
        # Show window values
        values_text = Text(f"Window: {int(window_values[:3])}", font_size=24)
        values_text2 = Text(f"        {int(window_values[3:6])}", font_size=24)
        values_text3 = Text(f"        {int(window_values[6:])}", font_size=24)
        
        values_group = VGroup(values_text, values_text2, values_text3)
        values_group.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        values_group.to_edge(DOWN, buff=1.5)
        
        self.play(Write(values_group))
        self.wait(0.5)
        
        # Sort and find median
        sorted_values = sorted(window_values)
        median_value = sorted_values[4]  # Middle value (5th element)
        
        sorted_text = Text(f"Sorted: {sorted_values}", font_size=24, color=BLUE)
        sorted_text.next_to(values_group, DOWN, buff=0.3)
        
        median_text = Text(f"Median = {median_value}", font_size=28, color=GREEN)
        median_text.next_to(sorted_text, DOWN, buff=0.3)
        
        self.play(Write(sorted_text))
        self.wait(0.3)
        self.play(Write(median_text))
        self.wait(0.5)
        
        # Update output grid
        output_data[row, col] = median_value
        output_idx = row * 5 + col
        
        new_cell = Square(side_length=0.6, stroke_width=2)
        intensity = median_value / 255
        new_cell.set_fill(WHITE, opacity=intensity * 0.5)
        
        value_text = Text(str(median_value), font_size=18)
        value_text.move_to(new_cell.get_center())
        new_cell.add(value_text)
        new_cell.move_to(output_grid[output_idx].get_center())
        
        self.play(
            Transform(output_grid[output_idx], new_cell),
            FadeOut(highlight),
            FadeOut(values_group),
            FadeOut(sorted_text),
            FadeOut(median_text)
        )
