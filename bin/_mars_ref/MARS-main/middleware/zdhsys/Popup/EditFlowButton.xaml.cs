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
using System.Windows.Shapes;
using zdhsys.Control;

namespace zdhsys.Popup
{
    /// <summary>
    /// EditFlowButton.xaml 的交互逻辑
    /// </summary>
    public partial class EditFlowButton : Window
    {
        public EditFlowButton(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
            DataToUI();
        }

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            isSave = true;
            Close();
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        public List<string> hexColor = new List<string>();

        private void DataToUI()
        {
            hexColor.Add("#944FFD");
            hexColor.Add("#FFBF16");
            hexColor.Add("#FF7089");
            hexColor.Add("#327FFE");
            hexColor.Add("#24282E");//黑
            hexColor.Add("#FFFFFF");//白

            _ = cbb_cmd.Items.Add("抓");
            _ = cbb_cmd.Items.Add("接");
            _ = cbb_cmd.Items.Add("放");
            _ = cbb_cmd.Items.Add("放+抓");
            cbb_cmd.SelectedIndex = 0;

            btnFlow.SetUI(hexColor[0], hexColor[0], hexColor[5], "设备名称");
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "确定");
            Btn_Del.SetUI("#FFFFFF", "#0066FF", "#0066FF", "删除节点");

            cb1.SetUI(hexColor[0]);
            cb2.SetUI(hexColor[1]);
            cb3.SetUI(hexColor[2]);
            cb4.SetUI(hexColor[3]);
            cb5.SetUI(hexColor[4]);

            cb6.SetUI(hexColor[4]);
            cb7.SetUI(hexColor[5]);

            cb1.Click += Cb_Click;
            cb2.Click += Cb_Click;
            cb3.Click += Cb_Click;
            cb4.Click += Cb_Click;
            cb5.Click += Cb_Click;

            cb6.Click += Cb6_Click;
            cb7.Click += Cb6_Click;

        }

        public bool isSave = false;
        public bool isDel = false;
        public int backIndex = 0;
        public int foreIndex = 5;

        private void Cb6_Click(object sender, RoutedEventArgs e)
        {
            cb6.SetIsCheck(false);
            cb7.SetIsCheck(false);
            ColorButton cb = sender as ColorButton;
            cb.SetIsCheck(true);

            if (cb.Name == "cb6")
            {
                btnFlow.SetForeColor(hexColor[4]);
                foreIndex = 4;
            }
            else if (cb.Name == "cb7")
            {
                btnFlow.SetForeColor(hexColor[5]);
                foreIndex = 5;
            }
        }

        public void SetUI(string hex1, string hex2, int index, string txt)
        {
            for (int i = 0; i < hexColor.Count; i++)
            {
                if (hexColor[i] == hex1)
                {
                    backIndex = i;
                    btnFlow.SetBackColor(hexColor[i]);
                    switch (i)
                    {
                        case 0:
                            cb1.SetIsCheck(true);
                            break;
                        case 1:
                            cb2.SetIsCheck(true);
                            break;
                        case 2:
                            cb3.SetIsCheck(true);
                            break;
                        case 3:
                            cb4.SetIsCheck(true);
                            break;
                        case 4:
                            cb5.SetIsCheck(true);
                            break;
                        default:
                            break;
                    }
                }
                if (hexColor[i] == hex2)
                {
                    btnFlow.SetForeColor(hexColor[i]);
                    foreIndex = i;
                    if (i == 4)
                    {
                        cb6.SetIsCheck(true);
                    }
                    else if(i == 5)
                    {
                        cb7.SetIsCheck(true);
                    }
                }
            }
            cbb_cmd.SelectedIndex = index;
            btnFlow.SetText(txt);
        }

        private void Cb_Click(object sender, RoutedEventArgs e)
        {
            cb1.SetIsCheck(false);
            cb2.SetIsCheck(false);
            cb3.SetIsCheck(false);
            cb4.SetIsCheck(false);
            cb5.SetIsCheck(false);
            ColorButton cb = sender as ColorButton;
            cb.SetIsCheck(true);

            switch (cb.Name)
            {
                case "cb1":
                    btnFlow.SetBackColor(hexColor[0]);
                    backIndex = 0;
                    break;
                case "cb2":
                    btnFlow.SetBackColor(hexColor[1]);
                    backIndex = 1;
                    break;
                case "cb3":
                    btnFlow.SetBackColor(hexColor[2]);
                    backIndex = 2;
                    break;
                case "cb4":
                    btnFlow.SetBackColor(hexColor[3]);
                    backIndex = 3;
                    break;
                case "cb5":
                    btnFlow.SetBackColor(hexColor[4]);
                    backIndex = 4;
                    break;
                default:
                    break;
            }
        }

        private void Btn_Del_Click(object sender, RoutedEventArgs e)
        {
            isDel = true;
            Close();
        }
        
        public int getCmd()
        {
            return cbb_cmd.SelectedIndex;
        }
    }
}
