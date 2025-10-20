"""
Main Gradio application for Width vs Depth visualization
"""

import pandas as pd

import gradio as gr

from sigmoid_plots import plot_neuron_sum, plot_three_graphs
from bump_plots import plot_single_bump, plot_bump_sum
from width_experiments import (
    plot_sigmoid_width_experiment,
    plot_relu_width_experiment,
    plot_bump_width_experiment
)
from training_plots import train_models


def create_app():
    """Create and configure the Gradio application"""
    
    with gr.Blocks(title="Width vs Depth - Comprehensive Analysis", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🧠 Width vs Depth: Neural Network Analysis
        
        This application visualizes the concepts of **width** and **depth** in neural networks 
        using bump, sigmoid, and ReLU functions.
        
        ## Mathematical Foundation:
        **Bump Function:** B(x; c, w, k) = σ(k(x - (c-w/2))) - σ(k(x - (c+w/2)))
        
        **Width:** Number of neurons per layer  
        **Depth:** Number of layers
        """)
        
        with gr.Tabs():
            # TAB 1: SIGMOID NEURON SUM
            with gr.Tab("🧠 Sigmoid Neuron Sum"):
                gr.Markdown("### Universal Approximation Theorem")
                with gr.Row():
                    with gr.Column(scale=1):
                        sig_num = gr.Slider(1, 20, value=1, step=1, label="🧠 Number of Neurons (N)")
                        sig_range = gr.Slider(1, 20, value=5, step=1, label="🔭 X Range")
                        sig_seed = gr.Slider(0, 100, value=42, step=1, label="🎲 Random Seed")
                        sig_show = gr.Checkbox(value=True, label="👁️ Show Individual Neurons")
                        
                    with gr.Column(scale=2):
                        sig_plot = gr.Plot(label="📊 Sigmoid Sum")
                
                sig_inputs = [sig_num, sig_range, sig_seed, sig_show]
                for inp in sig_inputs:
                    inp.change(fn=plot_neuron_sum, inputs=sig_inputs, outputs=sig_plot)
            
            # TAB 2: SIGMOID DIFFERENCE ANALYSIS
            with gr.Tab("🔬 Sigmoid Difference Analysis"):
                gr.Markdown("### Left Sigmoid - Right Sigmoid = Bump")
                with gr.Row():
                    with gr.Column(scale=1):
                        three_c = gr.Slider(-10, 10, value=0, step=0.1, label="📍 Center (c)")
                        three_w = gr.Slider(0.2, 10, value=4.0, step=0.1, label="📏 Width (w)")
                        three_k = gr.Slider(0.5, 20, value=5.0, step=0.1, label="🔪 Sharpness (k)")
                        three_a = gr.Slider(-3, 3, value=1.0, step=0.1, label="📈 Amplitude (α)")
                        three_r = gr.Slider(5, 20, value=10, step=1, label="🔭 X Range")
                        
                    with gr.Column(scale=2):
                        three_plot = gr.Plot(label="📊 3 Graph Analysis")
                
                three_inputs = [three_c, three_w, three_k, three_a, three_r]
                for inp in three_inputs:
                    inp.change(fn=plot_three_graphs, inputs=three_inputs, outputs=three_plot)
            
            # TAB 3: SINGLE BUMP
            with gr.Tab("🎯 Single Bump"):
                gr.Markdown("### Basic Bump Function")
                with gr.Row():
                    with gr.Column(scale=1):
                        single_c = gr.Slider(-10, 10, value=0, step=0.1, label="📍 Center (c)")
                        single_w = gr.Slider(0.1, 10, value=2.0, step=0.1, label="📏 Width (w)")
                        single_k = gr.Slider(0.1, 20, value=5.0, step=0.1, label="🔪 Sharpness (k)")
                        single_a = gr.Slider(-5, 5, value=1.0, step=0.1, label="📈 Amplitude (α)")
                        single_r = gr.Slider(5, 20, value=10, step=1, label="🔭 X Range")
                        
                    with gr.Column(scale=2):
                        single_plot = gr.Plot(label="📊 Single Bump")
                
                single_inputs = [single_c, single_w, single_k, single_a, single_r]
                for inp in single_inputs:
                    inp.change(fn=plot_single_bump, inputs=single_inputs, outputs=single_plot)
            
            # TAB 4: BUMP NEURON SUM
            with gr.Tab("🎯 Bump Neuron Sum"):
                gr.Markdown("### Local Feature Combinations")
                with gr.Row():
                    with gr.Column(scale=1):
                        bump_num = gr.Slider(1, 20, value=5, step=1, label="🎯 Number of Bumps (N)")
                        bump_range = gr.Slider(5, 20, value=10, step=1, label="🔭 X Range")
                        bump_seed = gr.Slider(0, 100, value=42, step=1, label="🎲 Random Seed")
                        bump_show = gr.Checkbox(value=True, label="👁️ Show Individual Bumps")
                        
                    with gr.Column(scale=2):
                        bump_plot = gr.Plot(label="📊 Bump Sum")
                
                bump_inputs = [bump_num, bump_range, bump_seed, bump_show]
                for inp in bump_inputs:
                    inp.change(fn=plot_bump_sum, inputs=bump_inputs, outputs=bump_plot)
            
            # TAB 5: SIGMOID WIDTH EXPERIMENT
            with gr.Tab("📊 Sigmoid Width Experiment"):
                gr.Markdown("""
                ### 🧠 Width Experiment with Sigmoid Neurons
                Visualize network capacity using different numbers of sigmoid neurons.
                
                **Formula:** f(x) = Σ αᵢ · σ(wᵢx + bᵢ)
                
                **Sigmoid Properties:**
                - Global activation (active across all x)
                - Smooth transitions
                - Standard MLP structure
                - Parameters: (w, b, α) × W = **3W params**
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sig_width_vals = gr.Textbox(
                            value="2, 4, 8, 16", 
                            label="🔢 Neuron Counts (comma-separated)"
                        )
                        sig_width_seed = gr.Slider(
                            0, 100, value=42, step=1, 
                            label="🎲 Random Seed"
                        )
                        sig_width_x_range = gr.Slider(
                            5, 20, value=10, step=1,
                            label="🔭 X Range"
                        )
                        sig_width_btn = gr.Button(
                            "🚀 Run Sigmoid Width Experiment", 
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 💡 Tips:
                        
                        **Few Neurons (2-4):**
                        - Simple, smooth functions
                        - Limited expressiveness
                        - Fast computation
                        
                        **Medium Neurons (8):**
                        - Complex patterns
                        - Multiple peaks/valleys
                        - Balanced performance
                        
                        **Many Neurons (16+):**
                        - Highly complex functions
                        - High capacity
                        - Overfitting risk
                        """)
                        
                    with gr.Column(scale=2):
                        sig_width_plot = gr.Plot(label="📊 Sigmoid Width Comparison")
                
                sig_width_inputs = [sig_width_vals, sig_width_seed, sig_width_x_range]
                sig_width_btn.click(
                    fn=plot_sigmoid_width_experiment, 
                    inputs=sig_width_inputs, 
                    outputs=sig_width_plot
                )
            
            # TAB 6: RELU WIDTH EXPERIMENT
            with gr.Tab("⚡ ReLU Width Experiment"):
                gr.Markdown("""
                ### ⚡ Width Experiment with ReLU Neurons
                Visualize piecewise linear approximation using different numbers of ReLU neurons.
                
                **Formula:** f(x) = Σ αᵢ · max(0, wᵢx + bᵢ)
                
                **ReLU Properties:**
                - Piecewise linear activation
                - Inactive below zero, linear above
                - Favorite of modern deep networks
                - Parameters: (w, b, α) × W = **3W params**
                - Easy gradient flow (no vanishing gradient)
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        relu_width_vals = gr.Textbox(
                            value="2, 4, 8, 16", 
                            label="🔢 Neuron Counts (comma-separated)"
                        )
                        relu_width_seed = gr.Slider(
                            0, 100, value=42, step=1, 
                            label="🎲 Random Seed"
                        )
                        relu_width_x_range = gr.Slider(
                            5, 20, value=10, step=1,
                            label="🔭 X Range"
                        )
                        relu_width_btn = gr.Button(
                            "🚀 Run ReLU Width Experiment", 
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 💡 Tips:
                        
                        **Few Neurons (2-4):**
                        - Coarse piecewise linear
                        - Sharp corners
                        - Simple approximation
                        
                        **Medium Neurons (8):**
                        - Finer segments
                        - Complex shapes
                        - Good balance
                        
                        **Many Neurons (16+):**
                        - Many piecewise linear segments
                        - Quasi-smooth appearance
                        - High capacity
                        
                        ---
                        ### 📚 ReLU Advantages:
                        
                        | Feature | ReLU |
                        |---------|------|
                        | **Computation** | Very fast |
                        | **Gradient** | Stable flow |
                        | **Sparse** | 50% zeros |
                        | **Usage** | Deep Networks |
                        """)
                        
                    with gr.Column(scale=2):
                        relu_width_plot = gr.Plot(label="📊 ReLU Width Comparison")
                        
                        gr.Markdown("""
                        ### 📖 Graph Explanation:
                        
                        Each subplot shows a different **W** (neuron count) value:
                        - **Green line**: Sum of all ReLU neurons
                        - **Sharp corners**: Piecewise linear structure
                        - **Asymmetric**: Activation in positive direction
                        
                        **Observations:**
                        - W=2: Few segments, coarse
                        - W=4: Medium detail
                        - W=8: Fine segments
                        - W=16: Quasi-smooth appearance
                        
                        **ReLU vs Sigmoid:**
                        - ReLU: Sharp, fast, sparse
                        - Sigmoid: Smooth, slow, dense
                        """)
                
                relu_width_inputs = [relu_width_vals, relu_width_seed, relu_width_x_range]
                relu_width_btn.click(
                    fn=plot_relu_width_experiment, 
                    inputs=relu_width_inputs, 
                    outputs=relu_width_plot
                )
            
            # TAB 7: BUMP WIDTH EXPERIMENT
            with gr.Tab("📊 Bump Width Experiment"):
                gr.Markdown("""
                ### 🎯 Width Experiment with Bump Neurons
                Visualize local feature combination using different numbers of bump neurons.
                
                **Formula:** f(x) = Σ αᵢ · B(x; cᵢ, wᵢ, kᵢ)
                
                **Bump Properties:**
                - Local activation (only in [c-w/2, c+w/2])
                - Localized features
                - Similar to RBF Networks
                - Parameters: (c, w, k, α) × W = **4W params**
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        bump_width_vals = gr.Textbox(
                            value="2, 4, 8, 16", 
                            label="🔢 Bump Counts (comma-separated)"
                        )
                        bump_width_seed = gr.Slider(
                            0, 100, value=42, step=1, 
                            label="🎲 Random Seed"
                        )
                        bump_width_x_range = gr.Slider(
                            5, 20, value=10, step=1,
                            label="🔭 X Range"
                        )
                        bump_width_btn = gr.Button(
                            "🚀 Run Bump Width Experiment", 
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 💡 Tips:
                        
                        **Few Bumps (2-4):**
                        - A few local peaks
                        - Simple, interpretable
                        - Each bump clearly visible
                        
                        **Medium Bumps (8):**
                        - Complex local structures
                        - Bumps start overlapping
                        - Interference effects
                        
                        **Many Bumps (16+):**
                        - Dense feature distribution
                        - Highly detailed functions
                        - High capacity
                        """)
                        
                    with gr.Column(scale=2):
                        bump_width_plot = gr.Plot(label="📊 Bump Width Comparison")
                
                bump_width_inputs = [bump_width_vals, bump_width_seed, bump_width_x_range]
                bump_width_btn.click(
                    fn=plot_bump_width_experiment, 
                    inputs=bump_width_inputs, 
                    outputs=bump_width_plot
                )
            
            # TAB 8: PYTORCH TRAINING
            with gr.Tab("🔥 PyTorch Training"):
                gr.Markdown("### 🎯 Training 4 Different Model Architectures")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ General Settings")
                        train_epochs = gr.Slider(10, 500, value=100, step=10, label="📈 Number of Epochs")
                        train_batch = gr.Slider(16, 256, value=64, step=16, label="📦 Batch Size")
                        train_samples = gr.Slider(500, 5000, value=1000, step=100, label="📊 Training Sample Count")
                        
                        gr.Markdown("---\n### 🏗️ Model 1: Wide-Shallow")
                        m1_layers = gr.Slider(1, 5, value=1, step=1, label="🔢 Number of Layers")
                        m1_neurons = gr.Slider(8, 256, value=128, step=8, label="🧠 Neurons/Layer")
                        m1_lr = gr.Slider(0.0001, 0.1, value=0.01, step=0.001, label="📉 Learning Rate")
                        m1_activation = gr.Radio(["sigmoid", "relu", "tanh"], value="relu", label="⚡ Activation")
                        
                        gr.Markdown("---\n### 🏗️ Model 2: Medium-Medium")
                        m2_layers = gr.Slider(1, 5, value=2, step=1, label="🔢 Number of Layers")
                        m2_neurons = gr.Slider(8, 256, value=64, step=8, label="🧠 Neurons/Layer")
                        m2_lr = gr.Slider(0.0001, 0.1, value=0.01, step=0.001, label="📉 Learning Rate")
                        m2_activation = gr.Radio(["sigmoid", "relu", "tanh"], value="relu", label="⚡ Activation")
                        
                        gr.Markdown("---\n### 🏗️ Model 3: Narrow-Deep")
                        m3_layers = gr.Slider(1, 5, value=3, step=1, label="🔢 Number of Layers")
                        m3_neurons = gr.Slider(8, 256, value=32, step=8, label="🧠 Neurons/Layer")
                        m3_lr = gr.Slider(0.0001, 0.1, value=0.01, step=0.001, label="📉 Learning Rate")
                        m3_activation = gr.Radio(["sigmoid", "relu", "tanh"], value="relu", label="⚡ Activation")
                        
                        gr.Markdown("---\n### 🏗️ Model 4: Very Narrow-Very Deep")
                        m4_layers = gr.Slider(1, 5, value=5, step=1, label="🔢 Number of Layers")
                        m4_neurons = gr.Slider(8, 256, value=16, step=8, label="🧠 Neurons/Layer")
                        m4_lr = gr.Slider(0.0001, 0.1, value=0.01, step=0.001, label="📉 Learning Rate")
                        m4_activation = gr.Radio(["sigmoid", "relu", "tanh"], value="relu", label="⚡ Activation")
                        
                        gr.Markdown("---")
                        train_btn = gr.Button("🚀 START TRAINING", variant="primary", size="lg")
                        train_status = gr.Textbox(label="📊 Status", value="Ready", interactive=False)
                    
                    with gr.Column(scale=2):
                        train_plot = gr.Plot(label="📈 Training Results")
                
                train_inputs = [
                    train_epochs, train_batch, train_samples,
                    m1_layers, m1_neurons, m1_lr, m1_activation,
                    m2_layers, m2_neurons, m2_lr, m2_activation,
                    m3_layers, m3_neurons, m3_lr, m3_activation,
                    m4_layers, m4_neurons, m4_lr, m4_activation
                ]
                
                train_btn.click(fn=train_models, inputs=train_inputs, outputs=[train_plot, train_status])
        
        gr.Markdown("""
        ---
        ## 🎓 User Guide
        
        **Recommended Order:**
        1. 🧠 Sigmoid Neuron Sum - Basic concepts
        2. 🔬 Sigmoid Difference Analysis - How bump is formed?
        3. 🎯 Single Bump - Detailed parameters
        4. 🎯 Bump Neuron Sum - Combinations
        5. 📊 Sigmoid Width Experiment - Global activation
        6. ⚡ ReLU Width Experiment - Piecewise linear
        7. 📊 Bump Width Experiment - Local activation
        8. 🔥 PyTorch Training - Real learning
        
        **💡 Key Concepts:**
        - **Width**: More neurons → Higher capacity
        - **Depth**: More layers → Hierarchical features
        - **Universal Approximation**: Any function can be approximated with sufficient neurons
        
        **🎯 Activation Comparison:**
        - **Sigmoid**: Smooth, global, bounded between 0-1
        - **ReLU**: Fast, sparse, piecewise linear
        - **Bump**: Local, interpretable, RBF-like
        
        ---
        **⚙️ Technical**: Python 3.x, NumPy, PyTorch, Plotly, Gradio
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Gradio application...")
    print("📊 Graphics Library: Plotly (Interactive)")
    print("✨ Features: Zoom, Pan, Hover, Toggle Legend")
    print("⚡ NEW: ReLU Width Experiment added!")
    
    demo = create_app()
    demo.launch(
        share=False, 
        server_port=7861,
        server_name="127.0.0.1",
        show_error=True,
        quiet=False
    )
