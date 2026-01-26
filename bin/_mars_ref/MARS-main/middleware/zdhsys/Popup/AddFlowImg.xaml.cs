using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using zdhsys.Bean;
using zdhsys.Control;
using zdhsys.entity;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddFlowImg.xaml 的交互逻辑
    /// </summary>
    public partial class AddFlowImg : Window
    {
        public AddFlowImg(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
        }

        private List<DeviceInfoModel> dims = null;

        public FlowModel Fm = null;
        public bool isUpdate = false;
        public bool isSave = false;

        private void DataToUI()
        {
            GridViewUnitls.AddRow(gdRight, 80);
            GridViewUnitls.AddColumn2(gdRight, gdRight.ActualWidth);
            FlowButton dev = new FlowButton()
            {
                Width = 120,
                Height = 80
            };
            dev.SetUI("#FFFFFF", "#0000FF", "#027AFF", "包硅反应");
            dev.Click += Dev_Click;
            Grid.SetRow(dev, gdRight.RowDefinitions.Count - 1);
            Grid.SetColumn(dev, 0);
            _ = gdRight.Children.Add(dev);

            GridViewUnitls.AddRow(gdCenter, gdCenter.ActualHeight);
            GridViewUnitls.AddColumn2(gdCenter, gdCenter.ActualWidth);

            gdCenter.MouseLeftButtonDown += gdCenter_MouseLeftButtonDown;
            gdCenter.MouseLeftButtonUp += GdCenter_MouseLeftButtonUp;

            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "保存");
            Btn_Close.SetUI("#FFFFFF", "#0066FF", "#0066FF", "取消");
            txt_z.SetUI("流程名称");
            txt_no.SetUI("流程编号");
            _ = cbb_Status.Items.Add("未执行");
            _ = cbb_Status.Items.Add("执行");
            cbb_Status.SelectedIndex = 0;

            dims = SqlHelper.GetDeviceInfo();
            for (int i = 0; i < dims.Count; i++)
            {
                AddLeft(dims[i]);
            }

            // 修改需要把数据重新还原出来
            if (isUpdate)
            {
                txt_z.SetText(Fm.FlowName);
                txt_no.SetText(Fm.FlowNO);
                cbb_Status.SelectedIndex = Fm.Status;
                List<FlowInfo> infos = JsonConvert.DeserializeObject<List<FlowInfo>>(Fm.FlowJson);
                for (int i = 0; i < infos.Count; i++)
                {
                    if (i < infos.Count - 1)
                    {
                        DeviceInfoModel dim = dims.Find(x => x.Id == infos[i].DeviceId);
                        FlowButton2 flow = new FlowButton2()
                        {
                            Width = 120,
                            Height = 80,
                            obj = dim,
                            ID = infos[i].FlowId,
                            CMD = infos[i].Cmd
                        };
                        flow.Click += Dev_Click1;
                        flow.SetUI(infos[i].tempHexBG, infos[i].tempHexBG, infos[i].tempHexTxt, infos[i].tempTxt);
                        Grid.SetRow(flow, gdCenter.RowDefinitions.Count - 1);
                        Grid.SetColumn(flow, 0);
                        _ = gdCenter.Children.Add(flow);
                        flow.SetTranslateTransform(new Point(infos[i].translateTransformX, infos[i].translateTransformY));
                    }
                    else if (i == infos.Count - 1)
                    {
                        DeviceInfoModel dim = dims.Find(x => x.Id == infos[i].DeviceId);
                        FlowButton2 flow = new FlowButton2()
                        {
                            Width = 120,
                            Height = 80,
                            obj = dim,
                            ID = infos[i].FlowId,
                            CMD = infos[i].Cmd
                        };
                        flow.Click += Dev_Click1;
                        flow.SetUI(infos[i].tempHexBG, infos[i].tempHexBG, infos[i].tempHexTxt, infos[i].tempTxt);
                        Grid.SetRow(flow, gdCenter.RowDefinitions.Count - 1);
                        Grid.SetColumn(flow, 0);
                        _ = gdCenter.Children.Add(flow);
                        flow.SetTranslateTransform(new Point(infos[i].translateTransformX, infos[i].translateTransformY));

                        DeviceInfoModel dim2 = dims.Find(x => x.Id == infos[i].DeviceId2);
                        FlowButton2 flow2 = new FlowButton2()
                        {
                            Width = 120,
                            Height = 80,
                            obj = dim2,
                            ID = infos[i].FlowId2,
                            CMD = infos[i].Cmd2
                        };
                        flow2.Click += Dev_Click1;
                        flow2.SetUI(infos[i].tempHexBG2, infos[i].tempHexBG2, infos[i].tempHexTxt2, infos[i].tempTxt2);
                        Grid.SetRow(flow2, gdCenter.RowDefinitions.Count - 1);
                        Grid.SetColumn(flow2, 0);
                        _ = gdCenter.Children.Add(flow2);
                        flow2.SetTranslateTransform(new Point(infos[i].translateTransformX2, infos[i].translateTransformY2));
                    }
                }
                //还原连接线
                for (int i = 0; i < infos.Count; i++)
                {
                    FlowLine f = new FlowLine
                    {
                        startFlow = FindFlowById(infos[i].FlowId),
                        startName = infos[i].flowDirName,
                        endFlow = FindFlowById(infos[i].FlowId2),
                        endName = infos[i].flowDirName2,
                        id = GlobalUnitils.GetNowTime(DateTime.Now) + i
                    };
                    fl.Add(f);
                }
                DataToUIFlowLine();
            }
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            DataToUI();
        }

        private void AddLeft(DeviceInfoModel dim)
        {
            GridViewUnitls.AddRow(gdLeft, 80);
            GridViewUnitls.AddColumn2(gdLeft, gdLeft.ActualWidth);
            FlowButton dev = new FlowButton()
            {
                Width = 120,
                Height = 80,
                obj = dim
            };
            dev.SetUI("#FFFFFF", "#027AFF", "#027AFF", dim.DeviceName);
            dev.Click += Dev_Click;
            Grid.SetRow(dev, gdLeft.RowDefinitions.Count - 1);
            Grid.SetColumn(dev, 0);
            _ = gdLeft.Children.Add(dev);
        }

        private void Dev_Click(object sender, RoutedEventArgs e)
        {
            FlowButton dev0 = sender as FlowButton;
            DeviceInfoModel dim = dev0.obj as DeviceInfoModel;
            FlowButton2 dev = new FlowButton2()
            {
                Width = 120,
                Height = 80,
                obj = dim,
                ID = GlobalUnitils.GetNowTime(DateTime.Now)
            };
            dev.Click += Dev_Click1;
            if (dim != null)
            {
                dev.SetUI("#FFFFFF", "#027AFF", "#027AFF", dim.DeviceName);
            }
            else
            {
                dev.SetUI("#FFFFFF", "#027AFF", "#027AFF", "包硅反应");
            }
            Grid.SetRow(dev, gdCenter.RowDefinitions.Count - 1);
            Grid.SetColumn(dev, 0);
            _ = gdCenter.Children.Add(dev);
        }

        private void Dev_Click1(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            EditFlowButton info = new EditFlowButton(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            FlowButton2 dev = sender as FlowButton2;
            info.SetUI(dev.tempHexBG, dev.tempHexTxt, dev.CMD, dev.tempTxt);
            _ = info.ShowDialog();
            // 这里要显示出来。不然会自动隐藏。有点没搞懂。
            dev.Visibility = Visibility.Visible;
            if (info.isSave)
            {
                dev.SetBackColor(info.hexColor[info.backIndex]);
                dev.SetForeColor(info.hexColor[info.foreIndex]);
                dev.CMD = info.getCmd();
            }
            else if (info.isDel)
            {
                Console.WriteLine($"删除相关的连线 {dev.tempTxt}");
                //1.删除相关的连线
                int index = findFlowIndex(dev, true);
                if (index != -1)
                {
                    DeleteLineByName("FL" + fl[index].id.ToString());
                    fl.RemoveAt(index);
                }
                index = findFlowIndex(dev, false);
                if (index != -1)
                {
                    DeleteLineByName("FL" + fl[index].id.ToString());
                    fl.RemoveAt(index);
                }
                //2.从gd里删除这个流程图
                DeleteFlowByName(dev.ID);
            }
        }

        #region 画线

        private bool isDragging = false;
        private FlowButton2 startFlow = null;
        private string startName;

        private void gdCenter_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            Point clickPoint = e.GetPosition(gdCenter);
            //Console.WriteLine($"flowButton 按下: {clickPoint.X},{clickPoint.Y}");
            FlowButton2 flowButton = getFlowByPoint(clickPoint);
            if (flowButton != null)
            {
                string name = flowButton.getEllipse();
                if (!string.IsNullOrEmpty(name))
                {
                    isDragging = true;
                    startFlow = flowButton;
                    startName = name;
                    //Console.WriteLine("按下：" + name);
                }
                //在 flowButton 上单击了鼠标左键
                //Console.WriteLine($"flowButton 按下: {clickPoint.X},{clickPoint.Y}");
            }
        }

        private void GdCenter_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (isDragging)
            {
                Point clickPoint = e.GetPosition(gdCenter);
                //Console.WriteLine($"flowButton 松开: {clickPoint.X},{clickPoint.Y}");
                FlowButton2 flowButton = getFlowByPoint(clickPoint);
                if (flowButton != null)
                {
                    string name = flowButton.getEllipse();
                    if (!string.IsNullOrEmpty(name))
                    {
                        //Console.WriteLine("松开：" + name);
                        FlowLine f = new FlowLine
                        {
                            startFlow = startFlow,
                            startName = startName,
                            endFlow = flowButton,
                            endName = name,
                            id = GlobalUnitils.GetNowTime(DateTime.Now)
                        };
                        //判断这个点是否已经有连接线。
                        int index = findFlowIndex(startFlow, true);
                        int index2 = findFlowIndex(flowButton, false);
                        if (index == -1 && index2 == -1)
                        {
                            fl.Add(f);
                        }
                        else
                        {
                            Console.WriteLine("连线重复了");
                        }
                    }
                }
            }
            isDragging = false;
            DataToUIFlowLine();
        }
        /// <summary>
        /// 保存所有连接线
        /// </summary>
        private List<FlowLine> fl = new List<FlowLine>();
        /// <summary>
        /// 绘制所有连接线
        /// </summary>
        private void DataToUIFlowLine()
        {
            for (int i = 0; i < fl.Count; i++)
            {
                Point p0 = new Point(0, 0);
                Point p1 = new Point(0, 0);
                FlowButton2 end = null;
                foreach (UIElement child in gdCenter.Children)
                {
                    if (child is FlowButton2 flow)
                    {
                        if (flow.ID == fl[i].startFlow.ID)
                        {
                            p0 = getFlowButtonLocation2(flow, flow.getStartPointByName(fl[i].startName));
                        }
                        else if (flow.ID == fl[i].endFlow.ID)
                        {
                            p1 = getFlowButtonLocation2(flow, flow.getStartPointByName(fl[i].endName));
                            end = flow;
                        }
                    }
                }
                if (p0.X == 0 && p0.Y == 0)
                {
                    fl.RemoveAt(i);
                    i--;
                    continue;
                }
                if (p1.X == 0 && p1.Y == 0)
                {
                    fl.RemoveAt(i);
                    i--;
                    continue;
                }
                if (Math.Abs(p0.X - p1.X) < 10)
                {
                    double ch = p0.X - p1.X;
                    if (end != null)
                    {
                        end.translateTransform.X += ch;
                    }
                    p1.X = p0.X;
                }
                else if (Math.Abs(p0.Y - p1.Y) < 10)
                {
                    double ch = p0.Y - p1.Y;
                    if (end != null)
                    {
                        end.translateTransform.Y += ch;
                    }
                    p1.Y = p0.Y;
                }
                AddArrowLineToGrid(p0, p1, "FL" + fl[i].id.ToString());
            }
        }

        private Point getFlowButtonLocation(FlowButton2 flowButton2)
        {
            Point topLeft = flowButton2.TranslatePoint(new Point(0, 0), gdCenter);
            Size size = new Size(flowButton2.ActualWidth, flowButton2.ActualHeight);
            Rect bounds = new Rect(topLeft, size);
            //Console.WriteLine($"FlowButton2位置：({bounds.X}, {bounds.Y}), 大小：({bounds.Width}, {bounds.Height})");
            Point p = flowButton2.getPoint();
            //Console.WriteLine($"flowButton2 getPoint={p.X},{p.Y}");
            Point p2 = flowButton2.getTranslateTransform();
            //Console.WriteLine($"flowButton2 getTranslateTransform={p2.X},{p2.Y}");
            Point p3 = new Point(bounds.X + p.X + p2.X, bounds.Y + p.Y + p2.Y);
            //Console.WriteLine($"相对于gdCenter的坐标 {p3.X},{p3.Y}");
            return p3;
        }

        private Point getFlowButtonLocation2(FlowButton2 flowButton2, Point p)
        {
            //Point topLeft = flowButton2.TranslatePoint(new Point(0, 0), gdCenter);
            Point topLeft = new Point(379, 309);
            Size size = new Size(120, 80);
            Rect bounds = new Rect(topLeft, size);
            //Console.WriteLine($"FlowButton2位置：({bounds.X}, {bounds.Y}), 大小：({bounds.Width}, {bounds.Height})");
            //Point p0 = flowButton2.getPoint();
            //Console.WriteLine($"flowButton2 getPoint={p0.X},{p0.Y}");
            Point p2 = flowButton2.getTranslateTransform();
            //Console.WriteLine($"flowButton2 getTranslateTransform={p2.X},{p2.Y}");
            Point p3 = new Point(bounds.X + p.X + p2.X, bounds.Y + p.Y + p2.Y);
            //Console.WriteLine($"相对于gdCenter的坐标 {p3.X},{p3.Y}");
            return p3;
        }

        private FlowButton2 getFlowByPoint(Point p)
        {
            UIElement element = gdCenter.InputHitTest(p) as UIElement;
            while (element != null)
            {
                if (element is FlowButton2)
                {
                    return element as FlowButton2;
                }
                element = VisualTreeHelper.GetParent(element) as UIElement;
            }
            return null;
        }


        /// <summary>
        /// 画线：Line
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <param name="name"></param>
        private void DrawLine(Point p1, Point p2, string name)
        {
            DeleteLineByName(name);
            Line myLine = new Line
            {
                Stroke = new SolidColorBrush(Color.FromArgb(255, 0, 255, 0)),
                StrokeThickness = 2,
                Name = name
            };
            myLine.X1 = p1.X;
            myLine.Y1 = p1.Y;
            myLine.X2 = p2.X;
            myLine.Y2 = p2.Y;
            Grid.SetRow(myLine, gdRight.RowDefinitions.Count - 1);
            Grid.SetColumn(myLine, 0);
            _ = gdCenter.Children.Add(myLine);
        }
        /// <summary>
        /// 添加画线，用path
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <param name="name"></param>
        private void DrawPathLine(Point p1, Point p2, string name)
        {
            Path myPath = new Path
            {
                Stroke = Brushes.Red,
                StrokeThickness = 2,
                Name = name,
                Data = new LineGeometry(p1, p2)
            };
            _ = gdCenter.Children.Add(myPath);
        }
        /// <summary>
        /// 添加画线且带箭头
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <param name="name"></param>
        private void AddArrowLineToGrid(Point p1, Point p2, string name)
        {
            DeleteLineByName(name);
            double startX = p1.X;
            double startY = p1.Y;
            double endX = p2.X;
            double endY = p2.Y;
            double arrowLength = 6;
            //Console.WriteLine($"p1({p1.X},{p1.Y}) p2({p2.X},{p2.Y})");
            // 计算箭头两个点的坐标
            double angle = Math.Atan2(endY - startY, endX - startX);
            double arrowX1 = endX - arrowLength * Math.Cos(angle - Math.PI / 6);
            double arrowY1 = endY - arrowLength * Math.Sin(angle - Math.PI / 6);
            double arrowX2 = endX - arrowLength * Math.Cos(angle + Math.PI / 6);
            double arrowY2 = endY - arrowLength * Math.Sin(angle + Math.PI / 6);

            double arrowX3 = endX - arrowLength * Math.Cos(angle);
            double arrowY3 = endY - arrowLength * Math.Sin(angle);

            // 创建箭头路径
            PathFigure pathFigure = new PathFigure
            {
                StartPoint = p2
            };
            pathFigure.Segments.Add(new LineSegment(new Point(arrowX1, arrowY1), true));
            pathFigure.Segments.Add(new LineSegment(new Point(arrowX2, arrowY2), true));
            pathFigure.IsClosed = true;

            PathFigure pathFigure2 = new PathFigure
            {
                StartPoint = p1
            };
            pathFigure2.Segments.Add(new LineSegment(new Point(arrowX3, arrowY3), true));

            PathGeometry pathGeometry = new PathGeometry();
            pathGeometry.Figures.Add(pathFigure);
            pathGeometry.Figures.Add(pathFigure2);

            // 创建 Path
            Path path = new Path
            {
                Stroke = Brushes.Red,
                StrokeThickness = 2,
                Fill = Brushes.Red,
                Data = pathGeometry,
                Name = name
            };
            _ = gdCenter.Children.Add(path);
        }

        /// <summary>
        /// 按名称删除画线
        /// </summary>
        /// <param name="name"></param>
        private void DeleteLineByName(string name)
        {
            for (int i = 0; i < gdCenter.Children.Count; i++)
            {
                UIElement child = gdCenter.Children[i];
                // 如果该子元素是线条（Line）类型，就将其从 myGrid 中删除
                if (child is Line)
                {
                    Line line = child as Line;
                    if (line.Name == name)
                    {
                        gdCenter.Children.Remove(child);
                        i--; // 修改循环索引，因为我们已经从集合中移除了一个元素
                    }
                }
                if (child is Path)
                {
                    Path line = child as Path;
                    if (line.Name == name)
                    {
                        gdCenter.Children.Remove(child);
                        i--; // 修改循环索引，因为我们已经从集合中移除了一个元素
                    }
                }
            }
        }
        /// <summary>
        /// 按ID删除流程图
        /// </summary>
        /// <param name="Id"></param>
        private void DeleteFlowByName(double Id)
        {
            for (int i = 0; i < gdCenter.Children.Count; i++)
            {
                UIElement child = gdCenter.Children[i];
                if (child is FlowButton2)
                {
                    FlowButton2 line = child as FlowButton2;
                    if (line.ID == Id)
                    {
                        gdCenter.Children.Remove(child);
                        i--; // 修改循环索引，因为我们已经从集合中移除了一个元素
                    }
                }
            }
        }

        /// <summary>
        /// 通过ID在grid里查找FlowButton2
        /// </summary>
        /// <param name="Id"></param>
        /// <returns></returns>
        private FlowButton2 FindFlowById(double Id)
        {
            for (int i = 0; i < gdCenter.Children.Count; i++)
            {
                UIElement child = gdCenter.Children[i];
                if (child is FlowButton2)
                {
                    FlowButton2 flow = child as FlowButton2;
                    if (flow.ID == Id)
                    {
                        return flow;
                    }
                }
            }
            return null;
        }

        #endregion

        /// <summary>
        /// 保存流程图
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            //1.要先对FL数组进行排序，排成一条线
            FlowButton2 startLine = null;
            List<FlowButton2> flows = new List<FlowButton2>();
            //1.1 先把所有连了线的设备拿出来。
            //1.2 这个设备如果不存在endFlow里面的话，说明它是第一个。
            for (int i = 0; i < fl.Count; i++)
            {
                if (!findFlow(flows, fl[i].startFlow))
                {
                    flows.Add(fl[i].startFlow);
                }
                if (!findFlow(flows, fl[i].endFlow))
                {
                    flows.Add(fl[i].endFlow);
                }
            }
            for (int i = 0; i < flows.Count; i++)
            {
                bool flag = true;
                for (int j = 0; j < fl.Count; j++)
                {
                    //在线的终点出现，就不是第一个。
                    if (flows[i].ID == fl[j].endFlow.ID)
                    {
                        flag = false;
                        break;
                    }
                }
                if (flag)
                {
                    startLine = flows[i];
                }
            }
            //2.找到第一个之后，就按顺序排下来就行了。
            List<FlowLine> fl2 = new List<FlowLine>();
            for (int i = 0; i < fl.Count; i++)
            {
                int index = findFlowIndex(startLine, true);
                fl2.Add(fl[index]);
                startLine = fl2[fl2.Count - 1].endFlow;
            }
            //Console.WriteLine("fl2.Count=" + fl2.Count);
            List<FlowInfo> infos = new List<FlowInfo>();
            for (int i = 0; i < fl2.Count; i++)
            {
                FlowInfo info = new FlowInfo();
                if (fl2[i].startFlow.obj != null)
                {
                    DeviceInfoModel dim = fl2[i].startFlow.obj as DeviceInfoModel;
                    info.DeviceId = dim.Id;
                }
                info.FlowId = fl2[i].startFlow.ID;
                info.flowDirName = fl2[i].startName;
                info.tempHexBd = fl2[i].startFlow.tempHexBd;
                info.tempHexBG = fl2[i].startFlow.tempHexBG;
                info.tempHexTxt = fl2[i].startFlow.tempHexTxt;
                info.tempTxt = fl2[i].startFlow.tempTxt;
                info.translateTransformX = fl2[i].startFlow.translateTransform.X;
                info.translateTransformY = fl2[i].startFlow.translateTransform.Y;
                info.Cmd = fl2[i].startFlow.CMD;

                if (fl2[i].endFlow.obj != null)
                {
                    DeviceInfoModel dim = fl2[i].endFlow.obj as DeviceInfoModel;
                    info.DeviceId2 = dim.Id;
                }
                info.FlowId2 = fl2[i].endFlow.ID;
                info.flowDirName2 = fl2[i].endName;
                info.tempHexBd2 = fl2[i].endFlow.tempHexBd;
                info.tempHexBG2 = fl2[i].endFlow.tempHexBG;
                info.tempHexTxt2 = fl2[i].endFlow.tempHexTxt;
                info.tempTxt2 = fl2[i].endFlow.tempTxt;
                info.translateTransformX2 = fl2[i].endFlow.translateTransform.X;
                info.translateTransformY2 = fl2[i].endFlow.translateTransform.Y;
                info.Cmd2 = fl2[i].endFlow.CMD;
                infos.Add(info);
                if (i < fl2.Count - 1)
                {
                    Console.Write($"  {fl2[i].startFlow.tempTxt}");
                }
                if (i == fl2.Count - 1)
                {
                    Console.WriteLine($"  {fl2[i].startFlow.tempTxt}  {fl2[i].endFlow.tempTxt}");
                }
            }
            //打印一下。
            //for (int i = 0; i < infos.Count; i++)
            //{
            //    Console.Write($"{infos[i].tempTxt}-{infos[i].flowDirName} 属性:{infos[i].tempHexBd},{infos[i].tempHexTxt},{infos[i].translateTransformX},{infos[i].translateTransformY}");
            //    Console.WriteLine($"  {infos[i].tempTxt2}-{infos[i].flowDirName2} 属性:{infos[i].tempHexBd2},{infos[i].tempHexTxt2},{infos[i].translateTransformX2},{infos[i].translateTransformY2}");
            //}
            string json = JsonConvert.SerializeObject(infos);
            Console.WriteLine("json=" + json);

            if (string.IsNullOrEmpty(txt_z.Text()) || string.IsNullOrEmpty(txt_no.Text()))
            {
                Point p = new Point(Left, Top);
                MessageInfo info = new MessageInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.SetUI("流程名称或流程编号不能为空!");
                _ = info.ShowDialog();
                return;
            }
            if (!isUpdate)
            {
                Fm = new FlowModel
                {
                    CreateTime = GlobalUnitils.GetNowTime(DateTime.Now)
                };
            }
            Fm.FlowName = txt_z.Text();
            Fm.FlowNO = txt_no.Text();
            Fm.Status = cbb_Status.SelectedIndex;
            Fm.FlowJson = json;
            isSave = true;
            Close();
        }

        private int findFlowIndex(FlowButton2 flow, bool frist)
        {
            if (flow == null)
            {
                return -1;
            }
            for (int i = 0; i < fl.Count; i++)
            {
                if (frist)
                {
                    if (fl[i].startFlow.ID == flow.ID)
                    {
                        return i;
                    }
                }
                else
                {
                    if (fl[i].endFlow.ID == flow.ID)
                    {
                        return i;
                    }
                }
            }
            Console.WriteLine($"未找到:{flow.tempTxt} frist={frist}");
            return -1;
        }

        private bool findFlow(List<FlowButton2> flows, FlowButton2 flow)
        {
            for (int i = 0; i < flows.Count; i++)
            {
                if (flows[i].ID == flow.ID)
                {
                    return true;
                }
            }
            return false;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
