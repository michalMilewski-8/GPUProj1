﻿namespace CPUVErsionTest1._0
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.drawing_panel = new System.Windows.Forms.PictureBox();
            this.cpu_use = new System.Windows.Forms.RadioButton();
            this.use_gpu = new System.Windows.Forms.RadioButton();
            ((System.ComponentModel.ISupportInitialize)(this.drawing_panel)).BeginInit();
            this.SuspendLayout();
            // 
            // drawing_panel
            // 
            this.drawing_panel.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.drawing_panel.BackColor = System.Drawing.Color.White;
            this.drawing_panel.Location = new System.Drawing.Point(2, 36);
            this.drawing_panel.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.drawing_panel.Name = "drawing_panel";
            this.drawing_panel.Size = new System.Drawing.Size(849, 561);
            this.drawing_panel.TabIndex = 0;
            this.drawing_panel.TabStop = false;
            this.drawing_panel.SizeChanged += new System.EventHandler(this.drawing_panel_SizeChanged);
            this.drawing_panel.Paint += new System.Windows.Forms.PaintEventHandler(this.drawing_panel_Paint);
            this.drawing_panel.MouseDown += new System.Windows.Forms.MouseEventHandler(this.drawing_panel_MouseDown);
            this.drawing_panel.MouseMove += new System.Windows.Forms.MouseEventHandler(this.drawing_panel_MouseMove);
            this.drawing_panel.MouseUp += new System.Windows.Forms.MouseEventHandler(this.drawing_panel_MouseUp);
            // 
            // cpu_use
            // 
            this.cpu_use.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.cpu_use.AutoSize = true;
            this.cpu_use.Location = new System.Drawing.Point(12, 12);
            this.cpu_use.Name = "cpu_use";
            this.cpu_use.Size = new System.Drawing.Size(53, 19);
            this.cpu_use.TabIndex = 1;
            this.cpu_use.Text = "CPU";
            this.cpu_use.UseVisualStyleBackColor = true;
            // 
            // use_gpu
            // 
            this.use_gpu.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.use_gpu.AutoSize = true;
            this.use_gpu.Checked = true;
            this.use_gpu.Location = new System.Drawing.Point(71, 12);
            this.use_gpu.Name = "use_gpu";
            this.use_gpu.Size = new System.Drawing.Size(54, 19);
            this.use_gpu.TabIndex = 1;
            this.use_gpu.TabStop = true;
            this.use_gpu.Text = "GPU";
            this.use_gpu.UseVisualStyleBackColor = true;
            this.use_gpu.CheckedChanged += new System.EventHandler(this.use_gpu_CheckedChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(852, 598);
            this.Controls.Add(this.use_gpu);
            this.Controls.Add(this.cpu_use);
            this.Controls.Add(this.drawing_panel);
            this.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.drawing_panel)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox drawing_panel;
        private System.Windows.Forms.RadioButton cpu_use;
        private System.Windows.Forms.RadioButton use_gpu;
    }
}

