using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;
using System;

namespace RTCK.NeuraBake.Runtime
{
    public class BakingCore
    {
        // RenderTexture Ç©ÇÁ Texture2D Ç÷ïœä∑Ç∑ÇÈÉÅÉ\ÉbÉh
        public static Texture2D ConvertRenderTextureToTexture2D(RenderTexture rt)
        {
            if (rt == null)
            {
                Debug.LogError("RenderTexture is null");
                return null;
            }

            var currentRT = RenderTexture.active;
            try
            {
                RenderTexture.active = rt;
                Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
                tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                tex.Apply();
                return tex;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error converting RenderTexture to Texture2D: {e.Message}");
                return null;
            }
            finally
            {
                RenderTexture.active = currentRT;
            }
        }

        // Texture2D Ç PNG ÉtÉ@ÉCÉãÇ∆ÇµÇƒï€ë∂Ç∑ÇÈÉÅÉ\ÉbÉh
        public static bool SaveTexture2DToPNG(Texture2D tex, string path)
        {
            if (tex == null)
            {
                Debug.LogError("Texture2D is null");
                return false;
            }

            try
            {
                byte[] pngData = tex.EncodeToPNG();
                if (pngData == null)
                {
                    Debug.LogError("Failed to encode texture to PNG");
                    return false;
                }

                File.WriteAllBytes(path, pngData);
                Debug.Log($"Texture saved to: {path}");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error saving PNG file: {e.Message}");
                return false;
            }
        }

        private readonly NeuraBakeSettings settings;
        private readonly Light[] sceneLights;
        private readonly NeuraBakeEmissiveSurface[] emissiveSurfaces; // î≠åıñ ÇÉLÉÉÉbÉVÉÖ
        private readonly Dictionary<Material, MaterialProperties> materialCache = new Dictionary<Material, MaterialProperties>();
        private readonly Dictionary<Mesh, MeshDataCache> meshDataCache = new Dictionary<Mesh, MeshDataCache>();
        private readonly object cacheLock = new object();

        private class MaterialProperties
        {
            public Material material; // å≥ÇÃÉ}ÉeÉäÉAÉãéQè∆Çï€éù
            public Color BaseColor { get; }
            public float Metallic { get; }
            public float Roughness { get; }
            public Texture2D BaseColorMap { get; }
            public Texture2D MetallicGlossMap { get; }
            public Texture2D NormalMap { get; }
            public Vector4 MainTexST { get; }
            public float Smoothness { get; }
            public Vector2 uvScale = Vector2.one;
            public Vector2 uvOffset = Vector2.zero;

            public MaterialProperties(Material material)
            {
                this.material = material;
                BaseColor = material.HasProperty("_BaseColor") ? material.GetColor("_BaseColor") :
                            (material.HasProperty("_Color") ? material.GetColor("_Color") : Color.white);
                Metallic = material.HasProperty("_Metallic") ? material.GetFloat("_Metallic") : 0f;
                Smoothness = material.HasProperty("_Glossiness") ? material.GetFloat("_Glossiness") :
                             (material.HasProperty("_Smoothness") ? material.GetFloat("_Smoothness") : 0.5f);
                Roughness = 1f - Smoothness;
                BaseColorMap = material.HasProperty("_MainTex") ? material.GetTexture("_MainTex") as Texture2D : null;
                MetallicGlossMap = material.HasProperty("_MetallicGlossMap") ? material.GetTexture("_MetallicGlossMap") as Texture2D : null;
                NormalMap = material.HasProperty("_BumpMap") ? material.GetTexture("_BumpMap") as Texture2D : null;
                MainTexST = material.HasProperty("_MainTex_ST") ? material.GetVector("_MainTex_ST") : new Vector4(1, 1, 0, 0);
                uvScale = new Vector2(MainTexST.x, MainTexST.y);
                uvOffset = new Vector2(MainTexST.z, MainTexST.w);
            }

            public Color GetSampledBaseColor(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * MainTexST.x + MainTexST.z, uv.y * MainTexST.y + MainTexST.w);
                Color texColor = BaseColorMap != null && BaseColorMap.isReadable ? BaseColorMap.GetPixelBilinear(tiledUv.x, tiledUv.y) : Color.white;
                return BaseColor * texColor;
            }

            public (float sampledMetallic, float sampledRoughness) GetSampledMetallicRoughness(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * MainTexST.x + MainTexST.z, uv.y * MainTexST.y + MainTexST.w);
                float finalMetallic = Metallic;
                float currentSmoothness = Smoothness;

                if (MetallicGlossMap != null && MetallicGlossMap.isReadable)
                {
                    Color metallicGlossSample = MetallicGlossMap.GetPixelBilinear(tiledUv.x, tiledUv.y);
                    bool isMaskMap = IsMaskMap(material, MetallicGlossMap);
                    if (isMaskMap)
                    {
                        finalMetallic = metallicGlossSample.a; // HDRP MaskMap: Metallic in A
                        currentSmoothness = metallicGlossSample.r; // HDRP MaskMap: Roughness in R
                    }
                    else
                    {
                        finalMetallic *= metallicGlossSample.r; // Standard shader: Metallic is in R channel
                        currentSmoothness *= metallicGlossSample.a; // Standard shader: Smoothness is in A channel
                    }
                }
                return (Mathf.Clamp01(finalMetallic), 1f - Mathf.Clamp01(currentSmoothness));
            }
        }

        private class MeshDataCache
        {
            public readonly Vector3[] Vertices;
            public readonly Vector3[] Normals;
            public readonly Vector2[] UVs;
            public readonly int[] Triangles;
            public MeshDataCache(Mesh mesh)
            {
                Vertices = mesh.vertices;
                Normals = mesh.normals;
                UVs = mesh.uv2 != null && mesh.uv2.Length == mesh.vertexCount ? mesh.uv2 : mesh.uv;
                Triangles = mesh.triangles;
            }
        }

        public BakingCore(NeuraBakeSettings bakeSettings)
        {
            settings = bakeSettings ?? throw new ArgumentNullException(nameof(bakeSettings));
            sceneLights = GameObject.FindObjectsOfType<Light>().Where(l => l.isActiveAndEnabled && l.intensity > 0).ToArray();
            emissiveSurfaces = GameObject.FindObjectsOfType<NeuraBakeEmissiveSurface>()
                                         .Where(es => es.enabled && es.bakeEmissive && es.intensity > 0)
                                         .ToArray();
        }

        private MaterialProperties GetMaterialProperties(Material material)
        {
            lock (cacheLock)
            {
                if (!materialCache.TryGetValue(material, out MaterialProperties props))
                {
                    props = new MaterialProperties(material);
                    materialCache[material] = props;
                }
                return props;
            }
        }

        private MeshDataCache GetMeshData(Mesh mesh)
        {
            lock (cacheLock)
            {
                if (!meshDataCache.TryGetValue(mesh, out MeshDataCache data))
                {
                    data = new MeshDataCache(mesh);
                    meshDataCache[mesh] = data;
                }
                return data;
            }
        }

        public async Task<Texture2D> BakeLightmapAsync(CancellationToken token, IProgress<(float percentage, string message)> progressReporter)
        {
            MeshRenderer[] renderers = GameObject.FindObjectsOfType<MeshRenderer>()
                                               .Where(r => r.enabled && r.gameObject.isStatic).ToArray();

            if (renderers.Length == 0)
            {
                progressReporter?.Report((1f, "ÉxÉCÉNëŒè€ÇÃê√ìIMeshRendererÇ™å©Ç¬Ç©ÇËÇ‹ÇπÇÒÅB"));
                return null;
            }

            MeshRenderer targetRenderer = renderers[0];
            Material targetMaterial = targetRenderer.sharedMaterial;
            MeshFilter meshFilter = targetRenderer.GetComponent<MeshFilter>();

            if (meshFilter == null || meshFilter.sharedMesh == null || targetMaterial == null)
            {
                progressReporter?.Report((1f, "ëŒè€ÉIÉuÉWÉFÉNÉgÇ…ÉÅÉbÉVÉÖÇ‹ÇΩÇÕÉ}ÉeÉäÉAÉãÇ™Ç†ÇËÇ‹ÇπÇÒÅB"));
                return null;
            }

            Mesh mesh = meshFilter.sharedMesh;
            MeshDataCache meshData = GetMeshData(mesh);

            if (meshData.UVs == null || meshData.UVs.Length == 0)
            {
                progressReporter?.Report((1f, "ëŒè€ÉÅÉbÉVÉÖÇ…ÉâÉCÉgÉ}ÉbÉvópÇÃUVÇ™Ç†ÇËÇ‹ÇπÇÒÅB"));
                return null;
            }

            int textureWidth = settings.atlasSize;
            int textureHeight = settings.atlasSize;
            Texture2D lightmapTexture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBAHalf, false, true);
            lightmapTexture.name = $"{targetRenderer.gameObject.name}_Lightmap_Baked";
            lightmapTexture.wrapMode = TextureWrapMode.Clamp;
            lightmapTexture.filterMode = FilterMode.Bilinear;

            Color[] pixels = new Color[textureWidth * textureHeight];
            int totalPixelsToProcess = textureWidth * textureHeight;
            int processedPixels = 0;
            float accumulatedBentNormalY = 0f;

            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    token.ThrowIfCancellationRequested();
                    Vector2 texelCenterUv = new Vector2((x + 0.5f) / textureWidth, (y + 0.5f) / textureHeight);

                    if (FindTriangleAndBarycentricCoords(meshData, texelCenterUv, out int triangleIndex, out Vector3 barycentricCoords))
                    {
                        int vIdx0 = meshData.Triangles[triangleIndex * 3 + 0];
                        int vIdx1 = meshData.Triangles[triangleIndex * 3 + 1];
                        int vIdx2 = meshData.Triangles[triangleIndex * 3 + 2];

                        Vector3 localPos = InterpolateVector3(meshData.Vertices[vIdx0], meshData.Vertices[vIdx1], meshData.Vertices[vIdx2], barycentricCoords);
                        Vector3 localNormal = InterpolateVector3(meshData.Normals[vIdx0], meshData.Normals[vIdx1], meshData.Normals[vIdx2], barycentricCoords).normalized;

                        Vector2 lightmapUV = InterpolateVector2(meshData.UVs[vIdx0], meshData.UVs[vIdx1], meshData.UVs[vIdx2], barycentricCoords);

                        Vector3 worldPos = targetRenderer.transform.TransformPoint(localPos);
                        Vector3 worldNormal = targetRenderer.transform.TransformDirection(localNormal).normalized;

                        Color accumulatedColor = Color.black;
                        accumulatedBentNormalY = 0f;
                        int validAoSamplesForBentNormal = 0;

                        for (int s = 0; s < settings.sampleCount; s++)
                        {
                            Vector2 currentSampleUV = texelCenterUv;
                            if (settings.sampleCount > 1)
                            {
                                float offsetX = (UnityEngine.Random.value - 0.5f) / textureWidth;
                                float offsetY = (UnityEngine.Random.value - 0.5f) / textureHeight;
                                currentSampleUV = new Vector2(texelCenterUv.x + offsetX, texelCenterUv.y + offsetY);
                            }

                            var colorInfo = CalculatePixelColor(worldPos, worldNormal, targetMaterial, lightmapUV, currentSampleUV);
                            accumulatedColor += colorInfo.color;
                            if (settings.directional && colorInfo.unoccludedRays > 0)
                            {
                                accumulatedBentNormalY += colorInfo.bentNormalY * colorInfo.unoccludedRays;
                                validAoSamplesForBentNormal += colorInfo.unoccludedRays;
                            }
                        }
                        Color finalPixelColor = accumulatedColor / settings.sampleCount;

                        if (settings.directional)
                        {
                            if (validAoSamplesForBentNormal > 0)
                            {
                                finalPixelColor.a = Mathf.Clamp01(0.5f + (accumulatedBentNormalY / validAoSamplesForBentNormal) * 0.5f);
                            }
                            else if (settings.useAmbientOcclusion && settings.aoSampleCount > 0)
                            {
                                finalPixelColor.a = 0f;
                            }
                            else
                            {
                                finalPixelColor.a = 0.5f;
                            }
                        }
                        else
                        {
                            finalPixelColor.a = 1.0f;
                        }

                        // ÉKÉìÉ}ï‚ê≥Ç™CalculatePixelColorÇÃíÜÇ≈ìKópÇ≥ÇÍÇƒÇ¢ÇÈÇÃÇ≈ÅAÇ±Ç±Ç≈ÇÕí«â¡ÇÃèàóùÇÕïsóv
                        pixels[y * textureWidth + x] = finalPixelColor;
                    }
                    else
                    {
                        pixels[y * textureWidth + x] = Color.clear;
                    }
                    processedPixels++;
                }
                if (y % 10 == 0 || y == textureHeight - 1)
                {
                    float percentage = (float)processedPixels / totalPixelsToProcess;
                    await Task.Yield();
                    progressReporter?.Report((percentage, $"ÉsÉNÉZÉãèàóùíÜ ({y + 1}/{textureHeight}çs)"));
                }
            }

            lightmapTexture.SetPixels(pixels);
            lightmapTexture.Apply(true, false);
            progressReporter?.Report((1f, "ÉâÉCÉgÉ}ÉbÉvê∂ê¨äÆóπ"));
            return lightmapTexture;
        }

        private bool FindTriangleAndBarycentricCoords(MeshDataCache meshData, Vector2 uv, out int triangleStartVertexIndex, out Vector3 barycentricCoords)
        {
            triangleStartVertexIndex = 0;
            barycentricCoords = Vector3.zero;
            for (int i = 0; i < meshData.Triangles.Length / 3; i++)
            {
                Vector2 uv0 = meshData.UVs[meshData.Triangles[i * 3 + 0]];
                Vector2 uv1 = meshData.UVs[meshData.Triangles[i * 3 + 1]];
                Vector2 uv2 = meshData.UVs[meshData.Triangles[i * 3 + 2]];
                barycentricCoords = CalculateBarycentricCoords(uv, uv0, uv1, uv2);
                if (barycentricCoords.x >= -1e-4f && barycentricCoords.y >= -1e-4f && barycentricCoords.z >= -1e-4f &&
                    barycentricCoords.x <= 1.0001f && barycentricCoords.y <= 1.0001f && barycentricCoords.z <= 1.0001f)
                {
                    triangleStartVertexIndex = i;
                    return true;
                }
            }
            return false;
        }

        private Vector3 CalculateBarycentricCoords(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            Vector2 v0 = b - a, v1 = c - a, v2 = p - a;
            float d00 = Vector2.Dot(v0, v0);
            float d01 = Vector2.Dot(v0, v1);
            float d11 = Vector2.Dot(v1, v1);
            float d20 = Vector2.Dot(v2, v0);
            float d21 = Vector2.Dot(v2, v1);
            float denom = d00 * d11 - d01 * d01;
            if (Mathf.Abs(denom) < 1e-6f) return new Vector3(-1, -1, -1);
            float v_ = (d11 * d20 - d01 * d21) / denom;
            float w_ = (d00 * d21 - d01 * d20) / denom;
            float u_ = 1.0f - v_ - w_;
            return new Vector3(u_, v_, w_);
        }

        private Vector3 InterpolateVector3(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 barycentric)
        {
            return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z;
        }

        private Vector2 InterpolateVector2(Vector2 v0, Vector2 v1, Vector2 v2, Vector3 barycentric)
        {
            return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z;
        }

        private (Color color, float bentNormalY, int unoccludedRays) CalculatePixelColor(
            Vector3 worldPos, Vector3 worldNormal, Material material,
            Vector2 lightmapUV,
            Vector2 materialSampleUV)
        {
            MaterialProperties matProps = GetMaterialProperties(material);
            Color albedo = matProps.GetSampledBaseColor(materialSampleUV);
            var (metallic, roughness) = matProps.GetSampledMetallicRoughness(materialSampleUV);

            Color finalColor = Color.black;

            Color skyContribution = CalculateSkyLight(worldPos, worldNormal, albedo);
            finalColor += skyContribution;

            foreach (Light light in sceneLights)
            {
                Vector3 lightDir;
                Color lightColorAtPoint = light.color * light.intensity;
                float attenuation = 1.0f;

                if (light.type == LightType.Directional)
                {
                    lightDir = -light.transform.forward;
                }
                else
                {
                    Vector3 lightToPointVec = worldPos - light.transform.position;
                    float distSqr = lightToPointVec.sqrMagnitude;
                    if (distSqr > light.range * light.range) continue;
                    float dist = Mathf.Sqrt(distSqr);
                    lightDir = -lightToPointVec.normalized;
                    attenuation = 1.0f / (1.0f + 25.0f * distSqr / (light.range * light.range));
                    if (light.type == LightType.Spot)
                    {
                        float spotAngle = Vector3.Angle(light.transform.forward, -lightDir);
                        if (spotAngle > light.spotAngle / 2f) continue;
                        float outerConeRad = (light.spotAngle / 2f) * Mathf.Deg2Rad;
                        float cosOuter = Mathf.Cos(outerConeRad);
                        float cosAngle = Vector3.Dot(light.transform.forward, -lightDir);
                        float spotFalloff = Mathf.Pow(Mathf.Clamp01((cosAngle - cosOuter) / (1f - cosOuter + 1e-4f)), 2f);
                        attenuation *= spotFalloff;
                    }
                }

                float shadowAccumulator = 0f;
                int actualShadowSamples = Mathf.Max(1, settings.shadowSamples);
                for (int i = 0; i < actualShadowSamples; i++)
                {
                    Vector3 jitteredLightDir = lightDir;
                    Vector3 shadowRayOrigin = worldPos + worldNormal * 0.001f;

                    if (actualShadowSamples > 1)
                    {
                        Vector3 randomOffset = UnityEngine.Random.onUnitSphere * 0.05f;
                        jitteredLightDir = (lightDir + randomOffset).normalized;
                    }

                    Ray shadowRay = new Ray(shadowRayOrigin, -jitteredLightDir);
                    float maxShadowDist = (light.type == LightType.Directional) ? 2000f : Vector3.Distance(worldPos, light.transform.position);
                    if (!Physics.Raycast(shadowRay, maxShadowDist - 0.002f))
                    {
                        shadowAccumulator += 1.0f;
                    }
                }
                float shadowFactor = shadowAccumulator / actualShadowSamples;

                if (shadowFactor > 0)
                {
                    Vector3 viewDir = worldNormal;
                    Color directLight = CalculatePBRDirectLight(
                        lightDir, viewDir, worldNormal,
                        lightColorAtPoint * attenuation * shadowFactor,
                        albedo, metallic, roughness
                    );
                    finalColor += directLight;
                }
            }

            foreach (NeuraBakeEmissiveSurface emissiveSurface in emissiveSurfaces)
            {
                MeshFilter emissiveMeshFilter = emissiveSurface.GetComponent<MeshFilter>();
                Renderer emissiveRenderer = emissiveSurface.GetComponent<Renderer>();
                if (emissiveMeshFilter == null || emissiveMeshFilter.sharedMesh == null || emissiveRenderer == null) continue;

                Mesh emissiveMesh = emissiveMeshFilter.sharedMesh;
                int emissiveSamplePoints = 16;

                for (int i = 0; i < emissiveSamplePoints; i++)
                {
                    int randomTriangleIndex = UnityEngine.Random.Range(0, emissiveMesh.triangles.Length / 3);
                    Vector3 b = GetRandomBarycentricCoords();

                    int vIdx0 = emissiveMesh.triangles[randomTriangleIndex * 3 + 0];
                    int vIdx1 = emissiveMesh.triangles[randomTriangleIndex * 3 + 1];
                    int vIdx2 = emissiveMesh.triangles[randomTriangleIndex * 3 + 2];

                    Vector3 emissivePointLocal = InterpolateVector3(emissiveMesh.vertices[vIdx0], emissiveMesh.vertices[vIdx1], emissiveMesh.vertices[vIdx2], b);
                    Vector3 emissiveNormalLocal = InterpolateVector3(emissiveMesh.normals[vIdx0], emissiveMesh.normals[vIdx1], emissiveMesh.normals[vIdx2], b).normalized;

                    Vector3 emissivePointWorld = emissiveSurface.transform.TransformPoint(emissivePointLocal);
                    Vector3 emissiveNormalWorld = emissiveSurface.transform.TransformDirection(emissiveNormalLocal).normalized;

                    Vector3 dirToEmissive = emissivePointWorld - worldPos;
                    float distToEmissiveSqr = dirToEmissive.sqrMagnitude;
                    float distToEmissive = Mathf.Sqrt(distToEmissiveSqr);
                    Vector3 normalizedDirToEmissive = dirToEmissive / distToEmissive;

                    float NdotL_emissive = Mathf.Max(0, Vector3.Dot(worldNormal, normalizedDirToEmissive));
                    float NdotL_source = Mathf.Max(0, Vector3.Dot(emissiveNormalWorld, -normalizedDirToEmissive));

                    if (NdotL_emissive > 0 && NdotL_source > 0)
                    {
                        Ray visibilityRay = new Ray(worldPos + worldNormal * 0.001f, normalizedDirToEmissive);
                        if (!Physics.Raycast(visibilityRay, distToEmissive - 0.002f))
                        {
                            Color emissiveLightColor = emissiveSurface.emissiveColor * emissiveSurface.intensity * settings.emissiveBoost;
                            float formFactorApprox = (NdotL_emissive * NdotL_source) / (Mathf.PI * distToEmissiveSqr + 1e-4f);
                            formFactorApprox /= emissiveSamplePoints;

                            Color contribution = (albedo / Mathf.PI) * emissiveLightColor * formFactorApprox;
                            finalColor += contribution;
                        }
                    }
                }
            }

            float occlusionFactor = 0f;
            float bentNormalYComponent = 0f;
            int unoccludedRayCount = 0;

            if (settings.useAmbientOcclusion && settings.aoSampleCount > 0)
            {
                (occlusionFactor, bentNormalYComponent, unoccludedRayCount) = CalculateAmbientOcclusionAndBentNormal(worldPos, worldNormal, settings.aoSampleCount);
                finalColor *= (1f - occlusionFactor);
            }
            else if (settings.directional)
            {
                bentNormalYComponent = worldNormal.y;
                unoccludedRayCount = 1;
            }

            // ÉJÉâÅ[ÉXÉyÅ[ÉXï‚ê≥ÇìKóp
            finalColor = ApplyColorSpaceCorrection(finalColor);

            return (finalColor, bentNormalYComponent, unoccludedRayCount);
        }

        private Vector3 GetRandomBarycentricCoords()
        {
            float r1 = UnityEngine.Random.value;
            float r2 = UnityEngine.Random.value;
            float sqrtR1 = Mathf.Sqrt(r1);
            return new Vector3(1.0f - sqrtR1, sqrtR1 * (1.0f - r2), sqrtR1 * r2);
        }

        private Color CalculateSkyLight(Vector3 worldPos, Vector3 worldNormal, Color albedo)
        {
            Color skyColor;
            
            if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Skybox && RenderSettings.skybox != null)
            {
                // ÉXÉJÉCÉ{ÉbÉNÉXÇ©ÇÁÇÃä¬ã´åıÇÃãﬂéó
                skyColor = RenderSettings.ambientSkyColor;
                // é¿ç€ÇÃÉXÉJÉCÉ{ÉbÉNÉXÉTÉìÉvÉäÉìÉOÇÕÇ±Ç±Ç≈ÇÕè»ó™ÇµÅAambientSkyColorÇégóp
            }
            else if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Trilight)
            {
                float dotUp = Vector3.Dot(worldNormal, Vector3.up);
                float upLerp = Mathf.Clamp01(dotUp);
                float downLerp = Mathf.Clamp01(-dotUp);
                skyColor = Color.Lerp(RenderSettings.ambientEquatorColor, RenderSettings.ambientSkyColor, upLerp);
                skyColor = Color.Lerp(skyColor, RenderSettings.ambientGroundColor, downLerp);
            }
            else if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Flat)
            {
                skyColor = RenderSettings.ambientLight;
            }
            else
            {
                // ÇªÇÃëºÇÃÉÇÅ[ÉhÇ≈ÇÃÉtÉHÅ[ÉãÉoÉbÉN
                skyColor = RenderSettings.ambientSkyColor;
            }
            
            // ÉJÉâÅ[ÉXÉyÅ[ÉXï‚ê≥ÇìKóp
            Color result = skyColor * albedo * settings.skyIntensity * RenderSettings.ambientIntensity;
            return ApplyColorSpaceCorrection(result);
        }

        private Color CalculatePBRDirectLight(Vector3 L, Vector3 V, Vector3 N, Color lightColor, Color albedo, float metallic, float roughness)
        {
            L = L.normalized; V = V.normalized; N = N.normalized;
            Vector3 H = (L + V).normalized;
            float NdotL = Mathf.Max(0f, Vector3.Dot(N, L));
            if (NdotL <= 0f) return Color.black;
            float NdotV = Mathf.Max(0f, Vector3.Dot(N, V));
            float alpha = roughness * roughness;
            Color F0 = Color.Lerp(new Color(0.04f, 0.04f, 0.04f), albedo, metallic);
            float D_GGX = GGX_Distribution(N, H, alpha);
            Color F_Schlick = Fresnel_Schlick(Mathf.Max(0f, Vector3.Dot(H, V)), F0);
            float G_SmithJointGGX = Smith_Visibility_JointGGX(N, V, L, alpha);
            Color specular = (D_GGX * F_Schlick * G_SmithJointGGX) / (4f * NdotL * NdotV + 1e-6f);
            Color kd = new Color(1f - metallic, 1f - metallic, 1f - metallic, 1f);
            Color diffuse = kd * albedo / Mathf.PI;
            return (diffuse + specular) * lightColor * NdotL;
        }

        private float GGX_Distribution(Vector3 N, Vector3 H, float alpha)
        {
            float NdotH = Mathf.Max(0f, Vector3.Dot(N, H));
            float alphaSq = alpha * alpha;
            float denom = NdotH * NdotH * (alphaSq - 1f) + 1f;
            return alphaSq / (Mathf.PI * denom * denom);
        }

        private Color Fresnel_Schlick(float cosTheta, Color F0)
        {
            return F0 + (new Color(1f - F0.r, 1f - F0.g, 1f - F0.b)) * Mathf.Pow(1f - cosTheta, 5f);
        }

        private float Smith_Visibility_JointGGX(Vector3 N, Vector3 V, Vector3 L, float alpha)
        {
            float NdotV = Mathf.Max(0f, Vector3.Dot(N, V));
            float NdotL = Mathf.Max(0f, Vector3.Dot(N, L));
            float k_direct = (alpha + 2f * Mathf.Sqrt(alpha) + 1f) / 8f;
            float G_V = NdotV / (NdotV * (1f - k_direct) + k_direct + 1e-5f);
            float G_L = NdotL / (NdotL * (1f - k_direct) + k_direct + 1e-5f);
            return G_V * G_L;
        }

        private (float occlusion, float bentNormalY, int unoccludedRays) CalculateAmbientOcclusionAndBentNormal(Vector3 position, Vector3 normal, int sampleCount)
        {
            if (sampleCount <= 0) return (0f, normal.y, 0);

            float occlusionFactor = 0f;
            Vector3 accumulatedUnoccludedNormal = Vector3.zero;
            int unoccludedRayCount = 0;
            float maxAoDistance = 2.0f;

            for (int i = 0; i < sampleCount; i++)
            {
                Vector3 randomDir = UnityEngine.Random.onUnitSphere;
                if (Vector3.Dot(randomDir, normal) < 0)
                {
                    randomDir = -randomDir;
                }

                Ray aoRay = new Ray(position + normal * 0.001f, randomDir);

                // ÊúÄÈÅ©Âåñ„Åï„Çå„Åü„É¨„Ç§„Ç≠„É£„Çπ„ÉàÂá¶ÁêÜÔºà„Ç≠„É£„ÉÉ„Ç∑„É•Âà©Áî®Ôºâ
                if (!raycastCache.Raycast(aoRay, maxAoDistance))
                {
                    lock (cacheLock)
                {
                    accumulatedUnoccludedNormal += randomDir;
                    unoccludedRayCount++;
                }
                else
                {
                    occlusionFactor += 1f;
                }
            }

            float finalOcclusion = (sampleCount > 0) ? occlusionFactor / sampleCount : 0f;
            float finalBentNormalY = normal.y;

            if (unoccludedRayCount > 0)
            {
                finalBentNormalY = (accumulatedUnoccludedNormal / unoccludedRayCount).normalized.y;
            }

            return (finalOcclusion, finalBentNormalY, unoccludedRayCount);
        }

        private static bool IsMaskMap(Material material, Texture2D texture)
        {
            if (texture == null) return false;

            bool isHDRP = material.shader.name.Contains("HDRP") ||
                          material.shader.name.Contains("High Definition");
            bool isURP = material.shader.name.Contains("URP") ||
                         material.shader.name.Contains("Universal");

            bool hasMaskMapProperty = material.HasProperty("_MaskMap");

            return isHDRP || hasMaskMapProperty ||
                   (texture.name.Contains("_MaskMap") && isURP);
        }

        private Color ApplyColorSpaceCorrection(Color color)
        {
            if (QualitySettings.activeColorSpace == ColorSpace.Gamma)
                return new Color(
                    Mathf.LinearToGammaSpace(color.r),
                    Mathf.LinearToGammaSpace(color.g),
                    Mathf.LinearToGammaSpace(color.b),
                    color.a
                );
            return color;
        }
    }
}