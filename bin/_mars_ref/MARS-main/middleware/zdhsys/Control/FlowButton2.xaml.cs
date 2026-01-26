using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using zdhsys.Unitils;

namespace zdhsys.Control
{
    /// <summary>
    /// DevButton.xaml 的交互逻辑
    /// </summary>
    public partial class FlowButton2 : UserControl
    {
        public FlowButton2()
        {
            InitializeComponent();
            //ID = GlobalUnitils.GetNowTime(DateTime.Now);
        }

        public event RoutedEventHandler Click;
        /// <summary>
        /// 绑定设备实体用
        /// </summary>
        public object obj;
        /// <summary>
        /// 流程ID。 因为会有设备重复的情况，不适合使用设备的ID做为唯一
        /// </summary>
        public long ID;
        //背景颜色
        public string tempHexBG = "";
        //边框颜色
        public string tempHexBd = "";
        //字体颜色
        public string tempHexTxt = "";
        //文本内容
        public string tempTxt = "";
        //操作指令
        public int CMD = 0;

        /// <summary>
        /// 设置按钮样式
        /// </summary>
        /// <param name="hexBG">按钮背景色</param>
        /// <param name="hexBd">按钮边框色</param>
        /// <param name="hexTxt">按钮文本颜色</param>
        /// <param name="txt">按钮文本</param>
        public void SetUI(string hexBG, string hexBd, string hexTxt, string txt)
        {
            tempHexBG = hexBG;
            tempHexBd = hexBd;
            tempHexTxt = hexTxt;
            tempTxt = txt;
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
        /// <summary>
        /// 设置背景色和边框颜色
        /// </summary>
        /// <param name="hexBG"></param>
        public void SetBackColor(string hexBG)
        {
            tempHexBG = hexBG;
            tempHexBd = hexBG;
            Color color = (Color)ColorConverter.ConvertFromString(hexBG);
            SolidColorBrush brush = new SolidColorBrush(color);
            bd.Background = brush;

            Color color1 = (Color)ColorConverter.ConvertFromString(hexBG);
            SolidColorBrush brush1 = new SolidColorBrush(color1);
            bd.BorderBrush = brush1;
        }
        /// <summary>
        /// 设置字体颜色
        /// </summary>
        /// <param name="hexTxt"></param>
        public void SetForeColor(string hexTxt)
        {
            Color color2 = (Color)ColorConverter.ConvertFromString(hexTxt);
            SolidColorBrush brush2 = new SolidColorBrush(color2);
            tbk.Foreground = brush2;
            tempHexTxt = hexTxt;
        }
        /// <summary>
        /// 设置显示文本内容
        /// </summary>
        /// <param name="txt"></param>
        public void SetText(string txt)
        {
            tbk.Content = txt;
        }

        /// <summary>
        /// 返回控件的内部坐标
        /// </summary>
        /// <returns></returns>
        public Point getPoint()
        {
            Point p = new Point(0, 0)
            {
                X = startPoint.X - translateTransform.X,
                Y = startPoint.Y - translateTransform.Y
            };
            return p;
        }
        /// <summary>
        /// 获取当前单击位置是否有画线的情况
        /// </summary>
        /// <returns></returns>
        public string getEllipse()
        {
            Point p = getPoint();
            for (int i = 0; i < ps.Count; i++)
            {
                if (ps[i].Data.FillContains(p))
                {
                    //Console.WriteLine($" ellipse name={ps[i].Name}");
                    return ps[i].Name;
                }
            }
            return "";
        }
        /// <summary>
        /// 返回当前控件的相对坐标
        /// </summary>
        /// <returns></returns>
        public Point getTranslateTransform()
        {
            return new Point(translateTransform.X, translateTransform.Y);
        }
        /// <summary>
        /// 设置当前控件的相对坐标
        /// </summary>
        /// <param name="p"></param>
        public void SetTranslateTransform(Point p)
        {
            translateTransform.X += p.X;
            translateTransform.Y += p.Y;
        }

        #region 拖动
        //是否拖拽
        private bool isDragging;
        /// <summary>
        /// 当前单击坐标
        /// </summary>
        public Point startPoint;
        /// <summary>
        /// 单击圆的位置名称
        /// </summary>
        public string clickName;
        /// <summary>
        /// 鼠标按下
        /// </summary>
        /// <param name="e"></param>
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            startPoint = e.GetPosition(this);
            bool flag = false;
            Point p = new Point(0, 0)
            {
                X = startPoint.X - translateTransform.X,
                Y = startPoint.Y - translateTransform.Y
            };
            //过滤掉4个圆。点击这4个圆，不移动当前控件。
            for (int i = 0; i < ps.Count; i++)
            {
                if (ps[i].Data.FillContains(p))
                {
                    flag = true;
                    break;
                }
            }
            if (!flag)
            {
                isDragging = true;
                _ = bd.CaptureMouse();
            }
        }
        /// <summary>
        /// 鼠标移动
        /// </summary>
        /// <param name="e"></param>
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            if (isDragging)
            {
                Point endPoint = e.GetPosition(this);
                double offsetX = endPoint.X - startPoint.X;
                double offsetY = endPoint.Y - startPoint.Y;

                translateTransform.X += offsetX;
                translateTransform.Y += offsetY;

                startPoint = endPoint;
            }
        }
        /// <summary>
        /// 鼠标松开
        /// </summary>
        /// <param name="e"></param>
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            isDragging = false;
            startPoint = e.GetPosition(this);
            bd.ReleaseMouseCapture();
        }

        #endregion
        /// <summary>
        /// 保存4个圆的路径。用于判断是否点击了这些圆，哪个圆
        /// </summary>
        private static List<Path> ps = new List<Path>();

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            ps.Add(AddEllipse("left", new Point(7, 40)));
            ps.Add(AddEllipse("top", new Point(60, 7)));
            ps.Add(AddEllipse("right", new Point(113, 40)));
            ps.Add(AddEllipse("bottom", new Point(60, 73)));
        }
        /// <summary>
        /// 返回左上右下的圆心坐标。可用于主界面画线的时候用到
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Point getStartPointByName(string name)
        {
            if (name == "left")
            {
                return new Point(7, 40);
            }
            else if (name == "top")
            {
                return new Point(60, 7);
            }
            else if (name == "right")
            {
                return new Point(113, 40);
            }
            else
            {
                return new Point(60, 73);
            }
        }
        /// <summary>
        /// 添加圆型路径
        /// </summary>
        /// <param name="name"></param>
        /// <param name="p"></param>
        /// <returns></returns>
        private Path AddEllipse(string name, Point p)
        {
            // 创建圆形路径
            Path path = new Path
            {
                Stroke = new SolidColorBrush(Color.FromArgb(255, 2, 122, 255)),
                StrokeThickness = 2,
                Fill = new SolidColorBrush(Color.FromArgb(255, 255, 255, 255))
            };
            EllipseGeometry ellipse = new EllipseGeometry
            {
                Center = p, // 设置圆心坐标
                RadiusX = 6, // 半径
                RadiusY = 6
            };
            path.Data = ellipse;
            path.Name = name;
            // 添加路径到 Grid
            _ = gd.Children.Add(path);
            return path;
        }
        /// <summary>
        /// 右键编辑
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void UserControl_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }
    }
}
