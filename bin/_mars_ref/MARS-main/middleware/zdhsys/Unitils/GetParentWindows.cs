using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace zdhsys.Unitils
{
    public class GetParentWindows
    {
        /// <summary>
        /// 获取当前用户控件所在的左上角坐标
        /// 用于给弹窗定位，只覆盖，当前自定义控件。不是当前软件，也不是全屏。
        /// </summary>
        /// <param name="win"></param>
        /// <returns></returns>
        public static Point GetPoint(UserControl win)
        {
            Point relativePoint = win.TransformToAncestor(Application.Current.MainWindow).Transform(new Point(0, 0));   // 获取相对于主窗体的位置
            // 获取主窗体的位置
            double mainWindowLeft = Application.Current.MainWindow.Left;
            double mainWindowTop = Application.Current.MainWindow.Top;
            // 计算控件相对于主窗体的位置
            double controlX = mainWindowLeft + relativePoint.X;
            double controlY = mainWindowTop + relativePoint.Y;
            return new Point(controlX, controlY);
            //当前主窗体
            //Window parentWindow = Window.GetWindow(this);
            //if (parentWindow != null)
            //{
            //    info.Left = parentWindow.Left;
            //    info.Top = parentWindow.Top;
            //    info.Width = parentWindow.ActualWidth;
            //    info.Height = parentWindow.ActualHeight;
            //}
        }

        public static Point GetPoint2(UserControl win)
        {
            GeneralTransform transform = win.TransformToAncestor(win.Parent as UIElement);
            Point position = transform.Transform(new Point(0, 0));
            Console.WriteLine($"position({position.X},{position.Y})");
            return position;
        }
    }
}
