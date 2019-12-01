namespace CPUVErsionTest1._0
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
            this.components = new System.ComponentModel.Container();
            this.drawing_panel = new System.Windows.Forms.PictureBox();
            this.cpu_use = new System.Windows.Forms.RadioButton();
            this.use_gpu = new System.Windows.Forms.RadioButton();
            this.label1 = new System.Windows.Forms.Label();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.button1 = new System.Windows.Forms.Button();
            this.textBox1 = new System.Windows.Forms.TextBox();
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
            this.drawing_panel.Margin = new System.Windows.Forms.Padding(2);
            this.drawing_panel.Name = "drawing_panel";
            this.drawing_panel.Size = new System.Drawing.Size(768, 768);
            this.drawing_panel.TabIndex = 0;
            this.drawing_panel.TabStop = false;
            this.drawing_panel.SizeChanged += new System.EventHandler(this.drawing_panel_SizeChanged);
            // 
            // cpu_use
            // 
            this.cpu_use.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.cpu_use.AutoSize = true;
            this.cpu_use.Location = new System.Drawing.Point(12, 12);
            this.cpu_use.Name = "cpu_use";
            this.cpu_use.Size = new System.Drawing.Size(47, 17);
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
            this.use_gpu.Size = new System.Drawing.Size(48, 17);
            this.use_gpu.TabIndex = 1;
            this.use_gpu.TabStop = true;
            this.use_gpu.Text = "GPU";
            this.use_gpu.UseVisualStyleBackColor = true;
            this.use_gpu.CheckedChanged += new System.EventHandler(this.use_gpu_CheckedChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(191, 12);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(35, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "label1";
            // 
            // timer1
            // 
            this.timer1.Interval = 50;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(286, 3);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(54, 28);
            this.button1.TabIndex = 3;
            this.button1.Text = "start";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(359, 10);
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(142, 20);
            this.textBox1.TabIndex = 4;
            this.textBox1.Text = "1000";
            this.textBox1.TextChanged += new System.EventHandler(this.textBox1_TextChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(784, 811);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.use_gpu);
            this.Controls.Add(this.cpu_use);
            this.Controls.Add(this.drawing_panel);
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "Form1";
            this.Text = "t";
            ((System.ComponentModel.ISupportInitialize)(this.drawing_panel)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox drawing_panel;
        private System.Windows.Forms.RadioButton cpu_use;
        private System.Windows.Forms.RadioButton use_gpu;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.TextBox textBox1;
    }
}

