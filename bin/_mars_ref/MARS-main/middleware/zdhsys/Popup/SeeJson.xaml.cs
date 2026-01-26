using JsonTool.ViewModel;
using Newtonsoft.Json;
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
using zdhsys.entity;

namespace zdhsys.Popup
{
    /// <summary>
    /// SeeJson.xaml 的交互逻辑
    /// </summary>
    public partial class SeeJson : Window
    {
        public JsonViewModel JsonViewDataContext
        {
            get { return (JsonViewModel)GetValue(JsonViewDataContextProperty); }
            set { SetValue(JsonViewDataContextProperty, value); }
        }

        // Using a DependencyProperty as the backing store for JsonViewDataContext.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty JsonViewDataContextProperty =
            DependencyProperty.Register("JsonViewDataContext", typeof(JsonViewModel), typeof(MainWindow), new PropertyMetadata(new JsonViewModel()));
        public SeeJson(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
            Btn_Close.Click += Btn_Close_Click;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        public List<List<List<PrintCmd>>> pc;

        private void MainWin_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_Close.SetUI("#E8F4FF", "#97D4FF", "\\Image\\语音关闭.png", "#1990FF", "关闭");
            string json = JsonConvert.SerializeObject(pc);
            jv.SetText(json);
        }
    }
}
