using JsonTool.Helper;
using JsonTool.Views;
using Microsoft.Win32;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
namespace JsonTool.ViewModel
{
    public class JsonViewModel : ViewModelBase
    {
        #region Fields
        private string _jsonFilePath;
        #endregion

        #region Propertites
        private string _jsonString = "{}";

        public string JsonString
        {
            get { return _jsonString; }
            set { _jsonString = value; OnPropertyChanged(()=>JsonString); }
        }


        private JToken _jToken;
        public JToken JToken
        {
            get { return _jToken; }
            set
            {
                _jToken = value; OnPropertyChanged(() => JToken);
                this.JsonList.Clear();
                this.JsonList.Add(_jToken);
            }
        }

        private bool _isEditMode = true;
        public bool IsEditMode
        {
            get { return _isEditMode; }
            set { _isEditMode = value; OnPropertyChanged(() => IsEditMode); }
        }


        private bool _isBusy;
        public bool IsBusy
        {
            get { return _isBusy; }
            set { _isBusy = value; OnPropertyChanged(() => IsBusy); }
        }


        private ObservableCollection<JToken> _jsonList;

        public ObservableCollection<JToken> JsonList
        {
            get
            {
                if (_jsonList == null) _jsonList = new ObservableCollection<JToken>();

                return _jsonList;
            }
            set { _jsonList = value; OnPropertyChanged(() => JsonList); }
        }


        #endregion

        #region Commands
        public ICommand NewCommand => new LambdaCommand((o) =>
        {
            IsBusy = true;
            try
            {
                this.JsonString="{}";
                this.JToken= JToken.Parse(JsonString);
                this._jsonFilePath="";
            }
            catch (Exception e)
            {
                MessageBox.Show($"Not possible to save changes:{Environment.NewLine}{e.ToString()}");
            }
            IsBusy = false;
        });

        public ICommand SaveCommand => new LambdaCommand((o) =>
        {
            IsBusy = true;
            try
            {
                if (string.IsNullOrEmpty(_jsonFilePath))
                {
                    var dlg = new SaveFileDialog();

                    var result = dlg.ShowDialog();
                    if (result != true)
                    {
                        return;
                    }

                    _jsonFilePath=dlg.FileName;
                }

                File.WriteAllText(_jsonFilePath, this.JsonString);
            }
            catch (Exception e)
            {
                MessageBox.Show($"Not possible to save changes:{Environment.NewLine}{e.ToString()}");
            }
            IsBusy = false;
        });

        public ICommand OpenFileCommand => new LambdaCommand((o) =>
        {
            IsBusy = true;
            try
            {
                var dlg = new OpenFileDialog();
                var result = dlg.ShowDialog();
                if (result == true)
                {
                    OpenFileContent(dlg.FileName);
                }
            }
            catch (Exception e)
            {
                MessageBox.Show($"Not possible to open file:{Environment.NewLine}{e.ToString()}");
            }
            IsBusy = false;
        });

        public ICommand CopyValueFromJPropertyCommand => new LambdaCommand((o) =>
        {
            if (o.IsNotEmpty() && o is JProperty)
            {
                var jp = o as JProperty;
                Clipboard.SetText(jp.Value.ToString());
            }
        });
        public ICommand CopyNameFromJPropertyCommand => new LambdaCommand((o) =>
        {
            if (o.IsNotEmpty() && o is JProperty)
            {
                var jp = o as JProperty;
                Clipboard.SetText(jp.Name);
            }
        });
        public ICommand EditValueFromJPropertyCommand => new LambdaCommand((o) =>
        {
            if (o.IsNotEmpty() && o is JProperty)
            {
                var jp = o as JProperty;
                var dlg = new ChangeValueWin(jp.Value.ToObject<object>());
                if (dlg.ShowDialog() == true)
                {
                    var js = this.JToken;
                    var token = js.SelectToken(jp.Path);
                    if (token != null)
                    {
                        token.Replace(JToken.FromObject(dlg.Value));
                    }
                    else
                    {
                        throw new Exception($"'{jp.Path}' not found...");
                    }

                    //             jp.Value = new JValue(dlg.Value);
                    JsonString = JToken.ToString();
                    JToken = JToken.Parse(JsonString);
                }
            }
        });

        public ICommand FromClipboardCommand => new LambdaCommand((o) =>
        {
            var content = System.Windows.Clipboard.GetText(TextDataFormat.Text);
            if (!string.IsNullOrWhiteSpace(content))
                SetJson(content);
        });


        public ICommand JsonCompactCommand => new LambdaCommand((o) =>
        {
            dynamic parsedJson = JsonConvert.DeserializeObject(JsonString);
            this.JsonString=  JsonConvert.SerializeObject(parsedJson, Formatting.None);
        });

        public ICommand JsoFormatCommand => new LambdaCommand((o) =>
        {
            dynamic parsedJson = JsonConvert.DeserializeObject(JsonString);
            this.JsonString=  JsonConvert.SerializeObject(parsedJson, Formatting.Indented);
        });

        #endregion

        public JsonViewModel()
        {
            _isBusy = false;
            _isEditMode = true;
            SetJson(this.JsonString);
        }

        public JsonViewModel(string json)
        {
            _isBusy = false;
            _isEditMode = true;
            SetJson(json);
        }

        public void OpenFileContent(string filePath)
        {
            if (!string.IsNullOrWhiteSpace(filePath) && File.Exists(filePath))
            {
                _jsonFilePath = filePath;
                var content = File.ReadAllText(filePath);
                SetJson(content, false);
            }
        }

        public void SetJson(string json, bool clearSourceFilePath = true)
        {
            if (clearSourceFilePath)
                _jsonFilePath = "";
            try
            {
                JToken = JToken.Parse(json);
                JsonString = JToken.ToString();
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                MessageBox.Show("无效的JSON格式数据");
            }
        }

    }
}
