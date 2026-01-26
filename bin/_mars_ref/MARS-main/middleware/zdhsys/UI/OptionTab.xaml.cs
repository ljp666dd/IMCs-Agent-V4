using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
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
using zdhsys.Bean;
using zdhsys.Control;
using zdhsys.Popup;
using zdhsys.Unitils;

namespace zdhsys.UI
{
    /// <summary>
    /// UCOption.xaml 的交互逻辑
    /// </summary>
    public partial class OptionTab : UserControl
    {
        public OptionTab()
        {
            InitializeComponent();

            btn_search.Click += Btn_search_Click;
            btn_clear.Click += Btn_clear_Click;
            pg.Click += Pg_Click;
        }
        /// <summary>
        /// 分页控件
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Pg_Click(object sender, RoutedEventArgs e)
        {
            Btn_search_Click(null, null);
        }
        /// <summary>
        /// 重置
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_clear_Click(object sender, RoutedEventArgs e)
        {
            txt_option.SetText("");
            dpStart.Text = "";
            dpEnd.Text = "";
        }
        /// <summary>
        /// 搜索
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_search_Click(object sender, RoutedEventArgs e)
        {
            int index = cbb_status.SelectedIndex;
            string start = "";
            if (!string.IsNullOrEmpty(dpStart.Text))
            {
                start = GlobalUnitils.GetNowTime(DateTime.Parse(dpStart.Text + " 00:00:00")).ToString();
            }
            string end = "";
            if (!string.IsNullOrEmpty(dpEnd.Text))
            {
                end = GlobalUnitils.GetNowTime(DateTime.Parse(dpEnd.Text + " 23:59:59")).ToString();
            }
            vs = SqlHelper.GetOptionsInfoBy(txt_option.Text(), index, start, end);
            //刷新数量
            pg.SetNum(vs.Count);
            DataToUI();
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("配方名称");

            _ = cbb_status.Items.Add("正常");
            _ = cbb_status.Items.Add("已作废");
            _ = cbb_status.Items.Add("全部");
            cbb_status.SelectedIndex = 0;

            btn_search.SetUI("#027AFF", "#027AFF", "\\Image\\搜索.png", "#FFFFFF", "搜索");
            btn_clear.SetUI("#FFFFFF", "#027AFF", "\\Image\\重置.png", "#027AFF", "重置");

            btn_add.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "创建");
            btn_del.SetUI("#F2F7FF", "#FF0000", "\\Image\\delete.png", "#FF0000", "批量删除");
            btn_import.SetUI("#E8F4FF", "#97D4FF", "\\Image\\import.png", "#1990FF", "导入模板");
            btn_model.SetUI("#E8F4FF", "#97D4FF", "\\Image\\model.png", "#1990FF", "编辑模板");

            //gd_content.MouseLeftButtonUp += Gd_content_MouseLeftButtonUp;
            Btn_search_Click(null, null);
        }

        private List<Options> vs = new List<Options>();
        private List<Options> vs2 = new List<Options>();

        /// <summary>
        /// 哪个单元内容被单击了。 -- 暂时没啥用。
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Gd_content_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            Grid grid = (Grid)sender;
            Point clickPosition = e.GetPosition(grid);

            int row = -1;
            int column = -1;

            for (int i = 0; i < grid.RowDefinitions.Count; i++)
            {
                if (clickPosition.Y >= grid.RowDefinitions[i].Offset && clickPosition.Y <= grid.RowDefinitions[i].ActualHeight + grid.RowDefinitions[i].Offset)
                {
                    row = i;
                    break;
                }
            }

            for (int i = 0; i < grid.ColumnDefinitions.Count; i++)
            {
                if (clickPosition.X >= grid.ColumnDefinitions[i].Offset && clickPosition.X <= grid.ColumnDefinitions[i].ActualWidth + grid.ColumnDefinitions[i].Offset)
                {
                    column = i;
                    break;
                }
            }

            // 输出被单击的行和列
            Console.WriteLine("被单击的行：{0}", row);
            Console.WriteLine("被单击的列：{0}", column);

            // 遍历并设置 Grid 的每一行的背景色
        }

        private void DataToUI()
        {
            gd_content.Children.Clear();
            gd_content.RowDefinitions.Clear();
            gd_content.ColumnDefinitions.Clear();
            //处理分页
            int limit = (pg.page - 1) * pg.getPageNum();
            int page = pg.getPageNum();
            vs2.Clear();
            for (int i = limit; i < limit + page; i++)
            {
                if (i + 1 > vs.Count) { break; }
                vs2.Add(vs[i]);
            }

            for (int i = 0; i < vs2.Count; i++)
            {
                DataToGridView(vs2[i],i);
            }
        }

        private void DataToGridView(Options opt, int index)
        {
            // 创建新的行- 固定行高:40
            GridViewUnitls.AddRow(gd_content);

            // 创建并添加新的列到新行
            for (int i = 0; i < gd_header.ColumnDefinitions.Count; i++)
            {
                GridViewUnitls.AddColumn(gd_content, gd_header, i);

                // 添加对应的控件到新的格子中
                if (i == 0)
                {
                    GridViewUnitls.AddCellCheck(gd_content, i, opt);
                }
                else if (i == 1)
                {
                    GridViewUnitls.AddCell(gd_content, i, index + 1 + "");
                }
                else if (i == 2)
                {
                    GridViewUnitls.AddCell(gd_content, i, opt.OptionName);
                }
                else if (i == 3)
                {
                    GridViewUnitls.AddCell(gd_content, i, GlobalUnitils.GetByLong2(opt.CreateTime));
                }

                else if (i == 4)
                {
                    if (opt.Status == 0)
                    {
                        GridViewUnitls.AddCellStatus(gd_content, i, "正常", GridViewUnitls.ColorHexNormal);
                    }
                    else
                    {
                        GridViewUnitls.AddCellStatus(gd_content, i, "已作废", GridViewUnitls.ColorHexBreak);
                    }
                }
                else if (i == 5)
                {
                    DevOperation dev = new DevOperation
                    {
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center,
                        obj = opt//绑定对象。用于详情，修改，删除之类的。
                    };
                    dev.Click_Details += Dev_Click_Details;
                    dev.Click_Update += Dev_Click_Update;
                    dev.Click_Delete += Dev_Click_Delete;
                    Grid.SetRow(dev, gd_content.RowDefinitions.Count - 1);
                    Grid.SetColumn(dev, i);
                    gd_content.Children.Add(dev);
                }
            }
            //每行再加一个分割线
            GridViewUnitls.AddSpLine(gd_content, gd_header);
        }
        /// <summary>
        /// 配方详情
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Dev_Click_Details(object sender, RoutedEventArgs e)
        {
            DevOperation dev = sender as DevOperation;
            Options df = dev.obj as Options;
            Point p = GetParentWindows.GetPoint(this);
            AddOptionNew win = new AddOptionNew(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            win.opt = df;
            win.isSee = true;
            win.isUpdate = true;
            _ = win.ShowDialog();
        }

        /// <summary>
        /// 单个删除
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Dev_Click_Delete(object sender, RoutedEventArgs e)
        {
            DevOperation dev = sender as DevOperation;
            Options df = dev.obj as Options;
            Point p = GetParentWindows.GetPoint(this);
            DeleteInfo info = new DeleteInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            info.SetUI("配方:" + df.OptionName);
            _ = info.ShowDialog();
            if (info.flagDelete)
            {
                SqlHelper.Delete(df);
                Btn_search_Click(null, null);
            }
        }
        /// <summary>
        /// 修改配方
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Dev_Click_Update(object sender, RoutedEventArgs e)
        {
            DevOperation dev = sender as DevOperation;
            Options df = dev.obj as Options;
            Point p = GetParentWindows.GetPoint(this);
            AddOptionNew win = new AddOptionNew(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            win.opt = df;
            win.isUpdate = true;
            _ = win.ShowDialog();
            //点击保存了。
            if (win.flag_Save)
            {
                //修改缓存列表和数据库
                SqlHelper.Update(win.opt);
                Btn_search_Click(null, null);
            }
        }


        /// <summary>
        /// 新增配方
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_add_Click(object sender, RoutedEventArgs e)
        {
            Point p = GetParentWindows.GetPoint(this);

            SelectFlowData2 sfd = new SelectFlowData2(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = sfd.ShowDialog();
            if (sfd.flag_Save)
            {
                AddOptionNew win = new AddOptionNew(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight,
                    fm = sfd.fm
                };
                _ = win.ShowDialog();
                if (win.flag_Save)
                {
                    SqlHelper.Add(win.opt);
                    Btn_search_Click(null, null);
                }
            }
        }

        /// <summary>
        /// 批量删除
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_del_Click(object sender, RoutedEventArgs e)
        {
            List<CheckBox> cbs = GridViewUnitls.FindCheckBoxesInGrid(gd_content);
            int num = 0;
            for (int i = 0; i < cbs.Count; i++)
            {
                if ((bool)cbs[i].IsChecked)
                {
                    num++;
                }
            }
            if (num > 0)
            {
                Point p = GetParentWindows.GetPoint(this);
                DeleteInfo info = new DeleteInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.SetUI(":" + num + "个配方");
                _ = info.ShowDialog();
                if (info.flagDelete)
                {
                    for (int i = 0; i < cbs.Count; i++)
                    {
                        if ((bool)cbs[i].IsChecked)
                        {
                            Options dgi = cbs[i].Tag as Options;
                            SqlHelper.Delete(dgi);
                        }
                    }
                    Btn_search_Click(null, null);
                }
            }
        }

        /// <summary>
        /// 导入配方
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_import_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Text Files (*.txt)|*.txt";
            if (openFileDialog.ShowDialog() == true)
            {
                string filePath = openFileDialog.FileName;
                // 在这里使用 filePath 处理所选文件
                Console.WriteLine("path=" + filePath);
                string ret = File.ReadAllText(filePath);
                Console.WriteLine(ret);
                if (string.IsNullOrEmpty(ret)) return;
                Point p = GetParentWindows.GetPoint(this);
                OptionNewModel win = new OptionNewModel(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                win.isUpdate = true;
                win.jsonOption = ret;
                _ = win.ShowDialog();
            }
        }

        /// <summary>
        /// 模板
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_model_Click(object sender, RoutedEventArgs e)
        {
            Point p = GetParentWindows.GetPoint(this);
            OptionNewModel win = new OptionNewModel(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = win.ShowDialog();
        }
        /// <summary>
        /// 全选/取消全选
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cb_All_Checked(object sender, RoutedEventArgs e)
        {
            bool flag = (bool)cb_All.IsChecked;
            GridViewUnitls.CheckBoxesInGrid(gd_content, flag);
        }
    }
}
