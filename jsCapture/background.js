var links = {}; //Store the stream link for every tabId.

// When activating a tab
chrome.webRequest.onBeforeRequest.addListener(function(details) {
  if (details.url.includes("playlist.m3u8")){
    var url = details.url;
    var road = getRoad(url);
    console.log(road);
    links[road] = url;
  }
},
{
  urls: ["https://*.streamlock.net/live/*"]
});


chrome.browserAction.onClicked.addListener(function() {
  console.log(links);
});


function getRoad(url) {
  // Extracts road's name
  return (url.split("/")[4].split(".")[0]);
};
