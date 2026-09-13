// react-plotly.js ships no type declarations, and neither does the cartesian
// bundle. The app only needs the Plotly namespace to exist loosely enough to type
// layouts and traces; pulling in @types/plotly.js would add a large dependency for
// very little benefit here.
declare module "react-plotly.js";
declare module "plotly.js-cartesian-dist-min";

declare namespace Plotly {
  type Data = any;
  type Layout = any;
  type Config = any;
}
