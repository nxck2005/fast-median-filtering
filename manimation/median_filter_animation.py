from manim import *

class MedianFilterAnimation(Scene):
    def construct(self):
        # Title
        title = Text("Median Filter (3x3 Kernel)", font_size=48).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Input Matrix (Image Patch)
        input_matrix_data = [
            [12, 15, 18, 20, 22],
            [25, 95, 30, 32, 35],
            [40, 42, 10, 48, 50],
            [55, 58, 60, 9, 65],
            [70, 72, 75, 78, 1]
        ]
        input_matrix = Matrix(input_matrix_data, h_buff=1.5).scale(0.8)
        input_label = Text("Input Image Patch", font_size=24).next_to(input_matrix, DOWN)
        input_group = VGroup(input_matrix, input_label).to_edge(LEFT, buff=1)

        # Output Matrix
        output_matrix_data = [["?" for _ in range(3)] for _ in range(3)] # Use '?' as placeholder
        output_matrix = Matrix(output_matrix_data, h_buff=1.5).scale(0.8)
        output_label = Text("Output Image", font_size=24).next_to(output_matrix, DOWN)
        output_group = VGroup(output_matrix, output_label).to_edge(RIGHT, buff=1)

        self.play(Create(input_group), Create(output_group))
        self.wait(1)

        # --- FIX 1: Correctly define the initial kernel box ---
        # We get all entries, then create a VGroup for the top-left 3x3
        entries = input_matrix.get_entries()
        initial_kernel_group = VGroup(
            entries[0], entries[1], entries[2],
            entries[5], entries[6], entries[7],
            entries[10], entries[11], entries[12]
        )
        kernel_box = SurroundingRectangle(
            initial_kernel_group,
            buff=0.2,
            color=YELLOW
        )
        # --- End Fix 1 ---

        # Create the kernel box on screen before the loop starts
        self.play(Create(kernel_box))
        self.wait(1)

        # --- FIX 2: Create a VGroup for calculation text ---
        # This VGroup will hold the "Window:", "Sorted:", and "Median:" text
        calc_group = VGroup(
            Text("Window:", font_size=24),
            Text("Sorted:", font_size=24),
            Text("Median:", font_size=24)
        ).arrange(DOWN, buff=0.5).to_edge(DOWN, buff=1)
        
        # Create empty VGroups to hold the dynamic values
        window_values_mob = VGroup().next_to(calc_group[0], RIGHT, buff=0.5)
        sorted_values_mob = VGroup().next_to(calc_group[1], RIGHT, buff=0.5)
        median_text_mob = VGroup().next_to(calc_group[2], RIGHT, buff=0.5)
        
        self.play(Write(calc_group))
        # --- End Fix 2 ---


        # Animation
        for i in range(3):
            for j in range(3):
                # Move kernel
                start_index = i * 5 + j
                
                # Get the VGroup of all entries in the current kernel window
                kernel_entries = VGroup()
                for row in range(3):
                    for col in range(3):
                        kernel_entries.add(input_matrix.get_entries()[start_index + row * 5 + col])

                self.play(kernel_box.animate.move_to(kernel_entries.get_center()))
                self.wait(0.5)

                # Extract values
                window_values = []
                for row in range(3):
                    for col in range(3):
                        window_values.append(input_matrix_data[i + row][j + col])
                
                # Update the "Window:" text
                new_window_values_mob = VGroup(*[Text(str(val), font_size=24) for val in window_values]).arrange(RIGHT, buff=0.5).move_to(window_values_mob.get_center())
                if i==0 and j==0: # First run, just create
                    self.play(Write(new_window_values_mob))
                else: # Subsequent runs, transform
                    self.play(Transform(window_values_mob, new_window_values_mob))
                window_values_mob = new_window_values_mob # Keep reference
                self.wait(0.5)

                # Sort values
                sorted_values = sorted(window_values)
                new_sorted_values_mob = VGroup(*[Text(str(val), font_size=24) for val in sorted_values]).arrange(RIGHT, buff=0.5).move_to(sorted_values_mob.get_center())
                
                # Animate the sorting
                self.play(Transform(window_values_mob, new_sorted_values_mob))
                sorted_values_mob = new_sorted_values_mob # Keep reference
                self.wait(0.5)

                # Find median
                median_value = sorted_values[4]
                median_box = SurroundingRectangle(sorted_values_mob[4], color=GREEN)
                new_median_text = Text(str(median_value), font_size=24).move_to(median_text_mob.get_center())
                
                self.play(Create(median_box), Write(new_median_text))
                median_text_mob = new_median_text # Keep reference
                self.wait(0.5)

                # --- FIX 3: Correctly update the output matrix ---
                # Get the placeholder mobject
                output_entry_placeholder = output_matrix.get_entries()[i * 3 + j]
                
                # Create the final text mobject
                new_entry_text = Text(str(median_value), font_size=24, color=GREEN).move_to(output_entry_placeholder.get_center())
                
                # Animate the median box flying to the target
                self.play(median_box.animate.move_to(output_entry_placeholder.get_center()))
                
                # Transform the placeholder '?' into the new text
                # and fade out the green box
                self.play(
                    Transform(output_entry_placeholder, new_entry_text),
                    FadeOut(median_box)
                )
                self.wait(0.5)
                # --- End Fix 3 ---

                # Cleanup for next iteration
                self.play(
                    FadeOut(window_values_mob),
                    FadeOut(sorted_values_mob),
                    FadeOut(median_text_mob)
                )
                # Re-create empty placeholders for the next loop
                window_values_mob = VGroup().next_to(calc_group[0], RIGHT, buff=0.5)
                sorted_values_mob = VGroup().next_to(calc_group[1], RIGHT, buff=0.5)
                median_text_mob = VGroup().next_to(calc_group[2], RIGHT, buff=0.5)

        # Fade out the kernel box at the end
        self.play(FadeOut(kernel_box), FadeOut(calc_group))
        self.wait(2)