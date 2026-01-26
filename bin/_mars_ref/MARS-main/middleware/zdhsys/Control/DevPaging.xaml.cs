using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace zdhsys.Control
{
    /// <summary>
    /// DevPaging.xaml 的交互逻辑
    /// 本控件为自定义分页控件
    /// </summary>
    public partial class DevPaging : UserControl
    {
        public DevPaging()
        {
            InitializeComponent();

            InitUI();
        }
        public event RoutedEventHandler Click;
        private void InitUI()
        {
            img_left.Source = new BitmapImage(new Uri(AppDomain.CurrentDomain.BaseDirectory + "\\Image\\arrowLeft.png")); ;
            img_right.Source = new BitmapImage(new Uri(AppDomain.CurrentDomain.BaseDirectory + "\\Image\\arrowRight.png"));

            cbb.Items.Add("10条/页");
            cbb.Items.Add("20条/页");
            cbb.Items.Add("30条/页");
            cbb.SelectedIndex = 0;
            cbb.SelectionChanged += Cbb_SelectionChanged;
            btn_1.Click += Btn_Click;
            btn_2.Click += Btn_Click;
            btn_3.Click += Btn_Click;
            btn_4.Click += Btn_Click;
            btn_5.Click += Btn_Click;
        }

        private void Btn_Click(object sender, RoutedEventArgs e)
        {
            Button btn = sender as Button;
            int p = int.Parse(btn.Content.ToString());
            page = p;
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }

        private void Cbb_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            SetNum(all_num);
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }

        /// <summary>
        /// 获取分页，每页显示数量
        /// </summary>
        /// <returns></returns>
        public int getPageNum()
        {
            if (cbb.SelectedIndex == 1)
            {
                return 20;
            }
            else if (cbb.SelectedIndex == 2)
            {
                return 30;
            }
            return 10;
        }

        /// <summary>
        /// 当前选中页--默认为第1页就算没数据也是。
        /// </summary>
        public int page = 1;

        /// <summary>
        /// 总页数
        /// </summary>
        public int all_page;

        /// <summary>
        /// 记录总条数
        /// </summary>
        public int all_num;

        /// <summary>
        /// 设置记录总数
        /// </summary>
        /// <param name="num"></param>
        public void SetNum(int num)
        {
            all_num = num;
            lb_all.Content = "共" + num + "条";
            int pageSingle = getPageNum();
            double d = num * 1.0 / pageSingle;
            int pg = (int)d;
            if (d > pg)
            {
                pg++;
            }
            all_page = pg;
            //先全部放开。
            btn_1.Visibility = Visibility.Visible;
            btn_2.Visibility = Visibility.Visible;
            btn_3.Visibility = Visibility.Visible;
            btn_4.Visibility = Visibility.Visible;
            btn_5.Visibility = Visibility.Visible;
            btn_1.Content = 1;
            btn_2.Content = 2;
            btn_3.Content = 3;
            btn_4.Content = 4;
            btn_5.Content = 5;
            if (pg == 0)
            {
                btn_1.Visibility = Visibility.Collapsed;
                btn_2.Visibility = Visibility.Collapsed;
                btn_3.Visibility = Visibility.Collapsed;
                btn_4.Visibility = Visibility.Collapsed;
                btn_5.Visibility = Visibility.Collapsed;
            }
            else if (pg == 1)
            {
                btn_2.Visibility = Visibility.Collapsed;
                btn_3.Visibility = Visibility.Collapsed;
                btn_4.Visibility = Visibility.Collapsed;
                btn_5.Visibility = Visibility.Collapsed;
            }
            else if (pg == 2)
            {
                btn_3.Visibility = Visibility.Collapsed;
                btn_4.Visibility = Visibility.Collapsed;
                btn_5.Visibility = Visibility.Collapsed;
            }
            else if (pg == 3)
            {
                btn_4.Visibility = Visibility.Collapsed;
                btn_5.Visibility = Visibility.Collapsed;
            }
            else if (pg == 4)
            {
                btn_5.Visibility = Visibility.Collapsed;
            }

            //这里多做一个判断，就是删除了之后，当数量已经不能支撑当前页数，就减1
            if (num == 0)
            {
                page = 1;
            }
            else
            {
                if (getPageNum() * (page - 1) >= num)
                {
                    if (page > 1)
                    {
                        page--;
                    }
                    //if (Click != null)
                    //{
                    //    Click.Invoke(this, null);
                    //}
                }
            }
        }

        private void btn_go_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(txt.Text))
            {
                return;
            }

            int p = int.Parse(txt.Text);
            // 如果超出范围，就显示当前页码
            if (p <= 0 || p > all_page)
            {
                txt.Text = page.ToString();
                return;
            }
            //设置当前页码
            page = p;

            if (all_page < 5)
            {
                return;
            }
            //处理跳转页码
            //判断页码是否在当前显示的范围。如果是的话，就不处理
            if (btn_1.Content.ToString() == p.ToString() 
                || btn_2.Content.ToString() == p.ToString()
                || btn_3.Content.ToString() == p.ToString()
                || btn_4.Content.ToString() == p.ToString()
                || btn_5.Content.ToString() == p.ToString())
            {
                return;
            }
            if (p + 2 <= all_page)
            {
                btn_3.Content = page;
                btn_4.Content = page + 1;
                btn_5.Content = page + 2;
            }
            else if (p + 1 == all_page)
            {
                btn_3.Content = page - 1;
                btn_4.Content = page;
                btn_5.Content = page + 1;
            }
            else if (p == all_page)
            {
                btn_3.Content = page - 2;
                btn_4.Content = page - 1;
                btn_5.Content = page;
            }
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }

        /// <summary>
        /// 只能输入整数
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void txt_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            if (!int.TryParse(e.Text, out _))
            {
                e.Handled = true; // 阻止输入
            }
        }

        private void btn_left_Click(object sender, RoutedEventArgs e)
        {
            if (page == 1)
            {
                return;//最小值了。
            }
            page--;
            txt.Text = page.ToString();
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
            flush();
        }

        private void flush()
        {
            //这里要判断一下，如果页面上没有这个页码的话，就刷新一下按钮的显示
            //判断页码是否在当前显示的范围。如果是的话，就不处理
            if (btn_1.Content.ToString() == page.ToString()
                || btn_2.Content.ToString() == page.ToString()
                || btn_3.Content.ToString() == page.ToString()
                || btn_4.Content.ToString() == page.ToString()
                || btn_5.Content.ToString() == page.ToString())
            {
                return;
            }
            if (page + 2 <= all_page)
            {
                btn_3.Content = page;
                btn_4.Content = page + 1;
                btn_5.Content = page + 2;
            }
            else if (page + 1 == all_page)
            {
                btn_3.Content = page - 1;
                btn_4.Content = page;
                btn_5.Content = page + 1;
            }
            else if (page == all_page)
            {
                btn_3.Content = page - 2;
                btn_4.Content = page - 1;
                btn_5.Content = page;
            }
        }

        private void btn_Right_Click(object sender, RoutedEventArgs e)
        {
            if (page == all_page)
            {
                return;
            }
            page++;
            txt.Text = page.ToString();
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
            flush();
        }
    }
}
