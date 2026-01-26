using ICSharpCode.AvalonEdit.Folding;
using JsonTool.Helper;
using Newtonsoft.Json;
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Xamasoft.JsonClassGenerator;
using Xamasoft.JsonClassGenerator.CodeWriters;
using JsonTool.ViewModel;

namespace JsonTool.Views
{
    /// <summary>
    /// JsonView.xaml 的交互逻辑
    /// </summary>
    public partial class JsonView : UserControl
    {
        public delegate void GeneratedHandler(string savePath);

        public event GeneratedHandler Generated;

        JsonViewModel ViewModel => this.DataContext as JsonViewModel;

        #region Dependency Propertites
        /// <summary>
        /// 生成后处理方式
        /// </summary>
        public GeneratedModes GeneratedMode
        {
            get { return (GeneratedModes)GetValue(GeneratedModeProperty); }
            set { SetValue(GeneratedModeProperty, value); }
        }

        // Using a DependencyProperty as the backing store for GeneratedMode.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty GeneratedModeProperty =
            DependencyProperty.Register("GeneratedMode", typeof(GeneratedModes), typeof(JsonView), new PropertyMetadata(GeneratedModes.Save));




        #region ShowFileToolBar
        public bool ShowFileToolBar
        {
            get { return (bool)GetValue(ShowFileToolBarProperty); }
            set { SetValue(ShowFileToolBarProperty, value); }
        }

        // Using a DependencyProperty as the backing store for ShowFileToolBar.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty ShowFileToolBarProperty =
            DependencyProperty.Register("ShowFileToolBar", typeof(bool), typeof(JsonView), new PropertyMetadata(true));
        #endregion





        #endregion

        public JsonView()
        {
            InitializeComponent();
        }

        private readonly static ICodeWriter[] CodeWriters = new ICodeWriter[] {
            new CSharpCodeWriter(),
             new JavaCodeWriter(),
            new VisualBasicCodeWriter(),
            new TypeScriptCodeWriter()         
        };

        FoldingManager foldingManager;
        BraceFoldingStrategy foldingStrategy;

        private void jsonView_Loaded(object sender, RoutedEventArgs e)
        {
            this.jsonEditor.TextChanged += Text_TextChanged;
            foldingManager = FoldingManager.Install(this.jsonEditor.TextArea);
            foldingStrategy = new BraceFoldingStrategy();
            foldingStrategy.UpdateFoldings(foldingManager, jsonEditor.Document);

            //cmbLanguage.ItemsSource=CodeWriters;
            //cmbLanguage.SelectedIndex=0;
        }

        public void SetText(string json)
        {
            jsonEditor.Text = json;

        }

        private void jsonView_Unloaded(object sender, RoutedEventArgs e)
        {
        
        }

        private void Text_TextChanged(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() => { 
            foldingStrategy.UpdateFoldings(foldingManager, jsonEditor.Document);
            });
        }

        private void jsonEditor_LostFocus(object sender, RoutedEventArgs e)
        {
           
        }

        private void jsonEditor_LostKeyboardFocus(object sender, KeyboardFocusChangedEventArgs e)
        {
            var vm = this.DataContext as JsonViewModel;
            if (vm == null) return;

            vm.SetJson(jsonEditor.Document.Text);
        }



        private JsonClassGenerator Prepare()
        {
            var vm = this.DataContext as JsonViewModel;

            dynamic parsedJson = JsonConvert.DeserializeObject(vm.JsonString);
            string json = JsonConvert.SerializeObject(parsedJson, Formatting.None);

            if (json == string.Empty || json=="{}")
            {
                MessageBox.Show("Json内容不能为空。");
                return null;
            }
          

            var gen = new JsonClassGenerator
            {
                Example = json,
                InternalVisibility = false,
                //CodeWriter = (ICodeWriter)cmbLanguage.SelectedItem,
                DeduplicateClasses = false,
                TargetFolder = "",
                UseProperties =true,
                MainClass = "Class1",
                SortMemberFields =true,
                UsePascalCase = true,
                UseNestedClasses = false,
                ApplyObfuscationAttributes = false,
                SingleFile = true,
                ExamplesInDocumentation = false,
                Namespace = null,
                NoHelperClass = true,
                SecondaryNamespace =  null,
            };
            gen.ExplicitDeserialization = false && gen.CodeWriter is CSharpCodeWriter;

            return gen;
        }

        string _lastSavePath = "";
        private void btnGenerate_Click(object sender, RoutedEventArgs e)
        {
            string savePath = "";
            var gen = Prepare();

            if (gen==null) return;

            if (GeneratedMode == GeneratedModes.Save)
            {
                var sfd = new System.Windows.Forms.FolderBrowserDialog();
                sfd.SelectedPath = _lastSavePath;             

                if (sfd.ShowDialog() != System.Windows.Forms.DialogResult.OK) return;
                savePath = sfd.SelectedPath;
            }
            else 
            {
                savePath = System.IO.Path.Combine( AppDomain.CurrentDomain.BaseDirectory,"Temp","Generate",DateTime.Now.ToString("yyyyMMddmmHHss"));
            }
            gen.TargetFolder = savePath;
            gen.FeedBack += (msg) => 
            {
                Console.WriteLine(msg);
            };
            gen.GenerateClasses();

            if (GeneratedMode == GeneratedModes.ReturnCode && Generated != null) 
            {
                Generated(savePath);
            }

        }
    }

    /// <summary>
    /// 生成完成后处理方式
    /// </summary>
    public enum GeneratedModes 
    {
        Save=0,
        ReturnCode=1
    }

}
