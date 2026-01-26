using System;
using System.Globalization;
using System.Windows.Data;
using Newtonsoft.Json.Linq;

namespace JsonTool.Converters.Json
{
    class JPropertyValueToString : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if(value is JProperty)
            {
                var jp = value as JProperty;
                if (jp.Value.Type != JTokenType.Object && jp.Value.Type != JTokenType.Array)
                    return jp.Value.ToString();
            }
            return null;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
