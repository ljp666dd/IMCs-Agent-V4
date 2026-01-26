using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace zdhsys.Control
{
    /// <summary>
    /// MyControl.xaml 的交互逻辑
    /// </summary>
    public partial class MyControl : UserControl
    {
        public MyControl()
        {
            InitializeComponent();
        }

        private bool isDragging;
        private Point startPoint;
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);

            isDragging = true;
            startPoint = e.GetPosition(this);

            myLabel.CaptureMouse();
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            if (isDragging)
            {
                //Point endPoint = e.GetPosition(this);
                Point endPoint = e.GetPosition((UIElement)Parent);
                //if (endPoint.X < 15 || endPoint.Y < 1) return;
                double offsetX = endPoint.X - startPoint.X;
                double offsetY = endPoint.Y - startPoint.Y;

                double left = Canvas.GetLeft(this) + translateTransform.X;
                double top = Canvas.GetTop(this) + translateTransform.Y;
                double width = ActualWidth;
                double height = ActualHeight;
                
                Console.WriteLine($"坐标: ({left}, {top}), 大小: {width} x {height}");
                if (left < 0)
                {
                    translateTransform.X += Math.Abs(left);
                    return;
                }
                if (top < -10)
                {
                    translateTransform.Y += Math.Abs(top) - 10;
                    return;
                }
                translateTransform.X += offsetX;
                translateTransform.Y += offsetY;
                Console.WriteLine("endPoint=" + endPoint.X + "," + endPoint.Y);
                startPoint = endPoint;
            }
        }

        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);

            isDragging = false;

            myLabel.ReleaseMouseCapture();
        }
    }
}
