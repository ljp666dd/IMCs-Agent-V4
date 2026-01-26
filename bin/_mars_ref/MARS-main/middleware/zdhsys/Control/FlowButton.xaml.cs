using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;

namespace zdhsys.Control
{
    /// <summary>
    /// DevButton.xaml 的交互逻辑
    /// </summary>
    public partial class FlowButton : UserControl
    {
        public FlowButton()
        {
            InitializeComponent();
        }

        public event RoutedEventHandler Click;
        public object obj;
        public long ID;

        /// <summary>
        /// 设置按钮样式
        /// </summary>
        /// <param name="hexBG">按钮背景色</param>
        /// <param name="hexBd">按钮边框色</param>
        /// <param name="hexTxt">按钮文本颜色</param>
        /// <param name="txt">按钮文本</param>
        public void SetUI(string hexBG, string hexBd, string hexTxt, string txt)
        {
            Color color = (Color)ColorConverter.ConvertFromString(hexBG);
            SolidColorBrush brush = new SolidColorBrush(color);
            bd.Background = brush;

            Color color1 = (Color)ColorConverter.ConvertFromString(hexBd);
            SolidColorBrush brush1 = new SolidColorBrush(color1);
            bd.BorderBrush = brush1;

            Color color2 = (Color)ColorConverter.ConvertFromString(hexTxt);
            SolidColorBrush brush2 = new SolidColorBrush(color2);
            tbk.Foreground = brush2;
            tbk.Content = txt;
        }


        private void UserControl_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }

        #region 拖动

        //private bool isDragging;
        //private Point startPoint;
        //protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        //{
        //    base.OnMouseLeftButtonDown(e);

        //    isDragging = true;
        //    startPoint = e.GetPosition(this);
        //    bd.CaptureMouse();
        //}

        //protected override void OnMouseMove(MouseEventArgs e)
        //{
        //    base.OnMouseMove(e);

        //    if (isDragging)
        //    {
        //        Point endPoint = e.GetPosition(this);
        //        double offsetX = endPoint.X - startPoint.X;
        //        double offsetY = endPoint.Y - startPoint.Y;

        //        translateTransform.X += offsetX;
        //        translateTransform.Y += offsetY;

        //        startPoint = endPoint;
        //    }
        //}

        //protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        //{
        //    base.OnMouseLeftButtonUp(e);

        //    isDragging = false;

        //    bd.ReleaseMouseCapture();
        //}

        #endregion
    }
}
