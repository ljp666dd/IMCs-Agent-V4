using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Animation;

namespace zdhsys.viewModel
{
    public class GLA : AnimationTimeline
    {
        public static readonly DependencyProperty FromProperty =
            DependencyProperty.Register("From", typeof(GridLength), typeof(GLA));

        public static readonly DependencyProperty ToProperty =
            DependencyProperty.Register("To", typeof(GridLength), typeof(GLA));

        public override Type TargetPropertyType
        {
            get { return typeof(GridLength); }
        }

        public override object GetCurrentValue(object defaultOriginValue, object defaultDestinationValue, AnimationClock animationClock)
        {
            if (From != null && To != null)
            {
                double fromValue = From.Value;
                double toValue = To.Value;

                double progress = animationClock.CurrentProgress.Value;
                double value = fromValue + (toValue - fromValue) * progress;

                return new GridLength(value);
            }
            else
            {
                return defaultOriginValue;
            }
        }

        protected override Freezable CreateInstanceCore()
        {
            return new GLA();
        }

        public GridLength From
        {
            get { return (GridLength)GetValue(FromProperty); }
            set { SetValue(FromProperty, value); }
        }

        public GridLength To
        {
            get { return (GridLength)GetValue(ToProperty); }
            set { SetValue(ToProperty, value); }
        }
    }
}
